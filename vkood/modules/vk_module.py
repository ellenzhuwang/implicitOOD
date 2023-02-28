# ------------------------------------------------------------------------------------
# VK-OOD
# Author: Ellen Wang
# ------------------------------------------------------------------------------------
# Modified from METER (https://github.com/zdou0830/METER)
# Copyright (c) 2021 Microsoft Corporation. All Rights Reserved.
# Licensed under MIT License(https://github.com/zdou0830/METER/blob/main/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from ViLT (https://github.com/dandelin/ViLT)
# Copyright 2021-present NAVER Corp. All Rights Reserved.
# Licensed under Apache 2.0(https://github.com/dandelin/ViLT/blob/master/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from CLIP(https://github.com/openai/CLIP)
# Copyright (c) 2021 OpenAI. All Rights Reserved.
# Licensed under MIT License(https://github.com/openai/CLIP/blob/main/LICENSE)
# ------------------------------------------------------------------------------------
# Modified from Swin-Transformer(https://github.com/microsoft/Swin-Transformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved.
# Licensed under MIT License(https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE)
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.mixture import GaussianMixture

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin, vk_utils
from . import heads, objectives
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel

class Concept_ood(nn.Module):
    """
    Consensus-level feature learning module .
    """
    def __init__(self, image_dim, embed_dim, no_imgnorm=False, ):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Concept_ood, self).__init__()

        self.no_imgnorm = no_imgnorm
        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed attribute to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        
        self.embedding_1 = nn.Sequential(self.fc1,nn.BatchNorm1d(embed_dim),nn.Tanh())
        self.embedding_2 = nn.Sequential(self.fc2,nn.BatchNorm1d(embed_dim),nn.Tanh())
        self.embedding_3 = nn.Sequential(self.fc3)
        
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.smooth_coef = 10


    def forward(self, emb_instance, concept_feature, input_modal, GT_label, GT_label_ratio):
        """
        Forward propagation.
        :param emb_instance: encoded images or text, shape: (batch_size, emb_dim)
        :param concept_feature: concept feature, shape: (att_num, emb_dim)
        :return: emb_concept: consensus-level feature
                 weights_u, weights_v: predicted concept score
        """
        W_s = self.embedding_1(concept_feature)  # (concept_num, emb_dim)

        W_v_m = self.embedding_2(emb_instance)   # (bs, emb_dim)
        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_s.size()[0], 1)   # (bs, att_num, emb_dim)

        h_s = W_s.mul(W_v_m)    # (bs, concept_num, emb_dim)

        a_s = self.embedding_3(h_s) # (bs, concept_num, 1)
        a_s = a_s.squeeze(2)        # (bs, concept_num)

        weights = self.softmax(a_s * self.smooth_coef)

        if input_modal == 'textual':

            GT_label_scale = self.softmax(GT_label * self.smooth_coef)
            weights_u = GT_label_ratio * GT_label_scale + (1 - GT_label_ratio) * weights
            concept_feature = vk_utils.l2norm(concept_feature)

            emb_concept = (weights_u.unsqueeze(2) * concept_feature).sum(dim=1)

            if not self.no_imgnorm:
                emb_concept = vk_utils.l2norm(emb_concept)
            return emb_concept, weights_u

        elif input_modal == 'visual':

            weights_v = weights
            concept_feature = vk_utils.l2norm(concept_feature)

            emb_concept = (weights_v.unsqueeze(2) * concept_feature).sum(dim=1)
            if not self.no_imgnorm:
                emb_concept = vk_utils.l2norm(emb_concept)
            return emb_concept, weights_v


class VKOOD(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.is_clip= (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        #self.V_Concept_ood = Concept_ood(config['input_image_embed_size'], config['input_image_embed_size'] )
        #self.V_Concept_ood.apply(objectives.init_weights)
        #self.T_Concept_ood = Concept_ood(config['input_text_embed_size'], config['input_text_embed_size'])
        #self.T_Concept_ood.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vk_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)

        
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        x_features = text_embeds.cpu().detach().numpy()
        text_mu =[]
        for a in range(x_features.shape[0]):
            mean = np.mean(x_features[a])
            mean_embed = np.full((1,768),mean)
            text_mu.append(mean_embed)

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_fit(text_embeds.cpu().numpy())
        prob = gmm.predict_proba(text_embeds.cpu().numpy())
        prob = prob[:, gmm.means_.argmin()]

        #ood_scores = [vk_utils.rescaled_GEM_score(text_embed,text_mu,phi=1) for text_embed in x_features]
        ood_scores = [vk_utils.rescaled_GEM_score(text_embed,gmm,phi=1) for text_embed in x_features]
        g_ij = []
        for i in range(len(ood_scores)):
            if np.mean(ood_scores[i]) >= 1e-5:
                g_ij.append(1)
            else:
                g_ij.append(0)
        g_ij = torch.Tensor(g_ij).to(device)
        text_embeds = torch.einsum('i,jkl->ikl', g_ij, text_embeds)
        
        image_embeds = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }


        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vk_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vk_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vk_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vk_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vk_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vk_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vk_utils.set_schedule(self)
