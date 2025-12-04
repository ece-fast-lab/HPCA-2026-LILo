# Step 1: Baseline
In this folder, we provide the script for collecting latency for pure CPU inference of the two models.

## How to run
`bash ./baseline.sh <your docker name>`

## Description

The script will:

(1) Start a docker container that includes all necessary code and scripts for CPU inference 

(2) Collect latency data for llama and deepseek models

(3) Output the latency data as `.log` files under `./results/` for later processing
