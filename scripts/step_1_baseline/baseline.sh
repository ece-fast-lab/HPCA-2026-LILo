#!/bin/bash
# Collecting data for baseline (no decompression)
# Usage: bash ./baseline.sh <docker name>
name=$1

docker run -dit --net host --privileged --name baseline_${name} -v ./:/root/mnt -v /storage3:/storage3 -v /storage2:/storage2 -v /storage:/storage qirongx2/lilo:baseline bash
docker exec baseline_${name} bash -ic "
    source /root/miniforge3/etc/profile.d/conda.sh;
    conda activate py310;
    cd llm;
    source ./tools/env_activate.sh inference;
    cd /root/llm/inference;
    bash ./run_llama.sh; # !run llama baseline
    bash ./run_ds.sh; # !run deepseek baseline
    # python /root/results/extract_data.py # !Extract latency and move it under current folder
"
# Cleanup
docker stop baseline_${name}
docker rm baseline_${name}
