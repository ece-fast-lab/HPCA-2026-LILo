#!/bin/bash
# Collecting data for storage-offload characterization with HuggingFace
# Usage: bash ./storage_offload.sh <docker name>
name=$1

docker run -dit --net host --privileged --name storage_offload_${name} \
 -v ./:/root/mnt -v /storage3:/storage3 -v /storage2:/storage2 -v /storage:/storage \
 qirongx2/lilo:storage_offload bash

docker exec storage_offload_${name} bash -ic "
    source /root/miniforge3/etc/profile.d/conda.sh;
    conda activate py310;
    cd llm;
    source ./tools/env_activate.sh inference;
    
"
# Cleanup
docker stop storage_offload_${name}
docker rm storage_offload_${name}
