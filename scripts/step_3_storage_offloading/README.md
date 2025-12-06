# Step 3: Characterizing the storage-offloading overhead

In this folder, we provide the script for characterizing the storage-offloading overhead with HuggingFace Accelerate.

## How to run

bash ./storage_offload.sh <your docker name>

## Description

The script will:

(1) Start a docker container that includes all necessary code and scripts for running storage-offloaded inference with HuggingFace Accelerate 

(2) Collect latency data under different amounts of data being offloaded to storage

(3) Output the latency data as .log files under ./results/ for later processing
