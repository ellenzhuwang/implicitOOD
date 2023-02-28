# VK-OOD: Differentiable Outlier Detection Enables Robust Deep Multimodal Analysis

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

