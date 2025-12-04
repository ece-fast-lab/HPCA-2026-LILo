# Step 0: Environment setup
In this folder, we provide the script for environment setup. It contains the following parts:
1. model preparation
2. docker image download
3. Fix CPU frequency and IAA configuration

## TL;DR
In experiment server: DS model is prepared under `/storage/qirongx2/Small-DS-R1/`, Llama model is prepared under `/storage2/qirongx2/Slim-Llama3.1-405B/`. 

Run `bash ./get_docker.sh` and `bash ./env_setup.sh`.

## 1. Model Preparation
**(1) Cropped model:**
As mentioned in the paper, there is a compatibility issue between storage offloading and IPEX. While we are actively working on providing the full system evaluation on storage offloading and decompression, this folder only contains the decompression part.

Instead of running the full-scale model, we crop the models to fit into one socket of system memory. We provide the summary on how we did that in the table below.

| Model         | Method     | Cropped script | Download |
|---------------|------------|----------------|------------------|
|Llama3.1-405B  | 126 decoder layers &rarr; 21 decoder layers| [crop_llama.py](./crop_llama.py) | [hf_slim_llama](huggingface_link)
|DeepSeek-R1    | 3 dense layers + 57 MoE layers &rarr; 1 dense layer + 19 MoE layers | [crop_ds.py](./crop_ds.py) | [hf_slim_ds](huggingface_link)

We pre-process the models and store them in the storage devices in experiment server.

**(2) Patching for MoE-based models:**
For MoE-based models like DeepSeek-R1, we randomize the expert selection of the router to simulate real expert selection in cropped model. For DeepSeek, the patch can be found at [deepseek.patch](./deepseek.patch).

We pre-process the models and store the script with model parameters.

## 2. Download docker images
We prepare the docker images for quick download and setup.
Run `bash ./get_docker.sh` to download necessary docker images.

## 3. Stablize Env
Run `bash ./env_setup.sh` to fix core frequency, setup IAAs, etc.. 
