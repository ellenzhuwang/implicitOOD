# VK-OOD: Differentiable Outlier Detection Enables Robust Deep Multimodal Analysis

We move this project to a new repo [here](https://github.com/ellenzhuwang/implicit_vkood). It is accepted by NeurIPS23. :new:

In this work, we propose an end-to-end vision and language model incorporating explicit knowledge graphs. We also introduce an interactive out-of-distribution (OOD) layer using implicit network operator. The layer is used to filter noise that is brought by external knowledge base. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-art results but with significantly fewer samples and training time.

<img width="2318" alt="pipline-poster" src="https://github.com/ellenzhuwang/implicitOOD/assets/10067151/48121aa2-1647-4f4a-bb52-e920a8c19572">



# Install

To create a conda enviroment:
```
$ conda create -n vk_ood python=3.8 pip
$ conda activate vk_ood
```
To install other requirements:
```
$ pip install -r requirements.txt
```
# Run

## Pre-train:
```
$ python train.py data_root=/dataset/pretrain num_gpus=8 num_nodes=1 task_mlm_itm_clip_bert per_gpu_batchsize=64 clip16 text_roberta image_size=244
```
## Fine-tune:

We show an example here : fine-tunning and evaluating on VQA tasks:
```
$ python train.py data_root=/dataset/vqa num_gpus=8 num_nodes=1task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=pretrain.ckpt clip16 text_roberta image_size=244 clip_randaug
```
We provide our VK-OOD-CLIP/16B-RoBERTa fine-tuned on VQAv2 checkpoint [here](https://drive.google.com/file/d/12HcGhMhAroAExCtjPHfQ9XC99Libeotx/view?usp=sharing)

## Evaluate:
```
$ python train.py data_root=/dataset/vqa num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=vqav2.ckpt clip16 text_roberta image_size=244 test_only=True
```
To get test-dev and test-std results, submit result json file /results/vqa_submit_ckpt.json to [eval.ai](https://eval.ai/challenge/830/overview).

# Citation
If you use this repo for your work, please consider citing our paper and staring this repo:
```
@article{wang2023differentiable,
  title={Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis},
  author={Wang, Zhu and Medya, Sourav and Ravi, Sathya N},
  journal={arXiv preprint arXiv:2302.05608},
  year={2023}
}
```
```
@article{wang2023implicit,
  title={Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis},
  author={Wang, Zhu and Medya, Sourav and Ravi, Sathya},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={13854--13872},
  year={2023}
}
```

