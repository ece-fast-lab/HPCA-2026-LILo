# Step 2: Inference with decompression

In this folder, we provide the script for collecting latency for LILo (inference + decompression) without storage-offloading.

## How to run
`bash ./decomp.sh <your docker name>`

## Description

The script will:

(1) Start a docker container that includes all necessary code and scripts running llama with decompression 

(2) Start a docker container that includes all necessary code and scripts running deepseek with decompression

(3) Output the latency data as `.log` files under `./results/` for later processing
