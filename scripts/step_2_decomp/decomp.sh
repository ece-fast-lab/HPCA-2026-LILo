#!/bin/bash
# Collecting data for decompression + inference
# Usage: bash ./baseline.sh <docker name>
name=$1

#--------------#
# Decomp Llama #
#--------------#
docker run -dit --net host --privileged --name decomp_llama_${name} \
 -v ./:/root/mnt -v /storage3:/storage3 -v /storage2:/storage2 -v /storage:/storage \
 qirongx2/lilo:llama bash

docker exec decomp_llama_${name} bash -ic "
    source /root/miniforge3/etc/profile.d/conda.sh;
    conda activate py310;
    cd llm;
    source ./tools/env_activate.sh inference;
    cd /root/thread_pool/;
    bash ./build.sh; # rebuild decompression module
    cd /root/llm/inference;
    bash ./run_llm.sh
"
# Cleanup
docker stop decomp_llama_${name}
docker rm decomp_llama_${name}

#--------------#
# Decomp DS    #
#--------------#
docker run -dit --net host --privileged --name decomp_ds_${name} \
 -v ./:/root/mnt -v /storage3:/storage3 -v /storage2:/storage2 -v /storage:/storage \
 qirongx2/lilo:ds bash

docker exec decomp_ds_${name} bash -ic "
    source /root/miniforge3/etc/profile.d/conda.sh;
    conda activate py310;
    cd llm;
    source ./tools/env_activate.sh inference;
    pip install --upgrade transformers==4.49.0;
    cd /root/thread_pool/;
    bash ./build.sh; # rebuild decompression module
    cd /root/llm/inference;
    bash ./run_llm.sh
"
# Cleanup
docker stop decomp_ds_${name}
docker rm decomp_ds_${name}
