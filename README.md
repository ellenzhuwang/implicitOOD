# VK-OOD: Differentiable Outlier Detection Enables Robust Deep Multimodal Analysis
In this work, we propose an end-to-end vision and language model incorporating explicit knowledge graphs. We also introduce an interactive out-of-distribution (OOD) layer using implicit network operator. The layer is used to filter noise that is brought by external knowledge base. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-art results but with significantly fewer samples and training time.

<img width="1503" alt="pipeline" src="https://user-images.githubusercontent.com/10067151/221752945-3e359d09-46f1-412d-8205-ce3093f566de.png">


# Install

To create a conda enviroment:
$ conda create -n vk_ood python=3.8 pip
$ conda activate vk_ood

To install other requiiments:
$ pip install -r requirements.txt

# Run

We show an example here : fine-tunning and evaluating on VQA tasks:

## Fine-tuning:

$ python train.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
We provide our VK-OOD-VIT/16B-RoBERTa fine-tuned on VQAv2 checkpoint here: https://drive.google.com/file/d/12HcGhMhAroAExCtjPHfQ9XC99Libeotx/view?usp=sharing

## Evaluate:

$ python train.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True

To get test-dev and test-std results, submit result json file /results/vqa_submit_ckpt.json to eval.ai:https://eval.ai/challenge/830/overview.

