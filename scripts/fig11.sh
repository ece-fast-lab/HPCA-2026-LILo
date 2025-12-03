#!/bin/bash
# Baseline
docker run -dit --net host --privileged --name baseline -v /storage3:/storage3 -v /storage2:/storage2 -v /storage:/storage qirongx2/ipex-llm:main bash
docker exec baseline bash -lc "cd llm && source ./tools/env_activate.sh inference"
