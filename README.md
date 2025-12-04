# LILO: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration

This is the respository that holds the artifacts of HPCA 2026 --  LILO: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration

## System Prerequisite
**Hardware**
- A server with an Intel 4th generation Xeon Scalable Processor equipped with at least one IAA (4 IAAs is preferred to reproduce performance).
- Make sure `hardware prefetch`, `LLC prefetch` is turned on in BIOS.
- Recommend to have at least 1.2TB stoarge space for storing all models' parameters. 

**Software**
- Ubuntu 22.04 LTS
- Linux kernel: 6.8.0-49-generic
- gcc 13.1.0
- IAA software library: [Query Processing Library (QPL)](https://github.com/intel/qpl) (online [doc](https://intel.github.io/qpl/))
- IAA configuration library: [idxd-config](https://github.com/intel/idxd-config)

**Other**
- Turn on IOMMU in grub to use IAA: `GRUB_CMDLINE_LINUX="quiet iommu=pt intel_iommu=on sm_on no5lvl splash intel_pstate=disable efi=nosoftreserve nokaslr"`


## Reproduce results
1. Inference throughput and latency on DeepSeek-R1, Llama3-405B, Qwen3-235B, and OPT-175B (Fig.11, Fig.13)
2. Energy effciency (Fig.14)

## Reproduce procedure
0. Prepare model parameters and software environment 

    &rarr; [Step0 - environment setup](./scripts/step_0_env_setup/README.md)

1. Collecting latency results for baseline (only CPU inference latency)

    &rarr; [Step1 - Baseline](./scripts/step_1_baseline/)

2. Collecting latency results with inference + decompression 

    &rarr; [Step2 - Decompression](./scripts/step_2_decomp/)

3. Collecting latency results with inference + storage offloading
    
    &rarr; [Step3 - Storage offloading](./scripts/step_3_storage_offloading/)


## Estimated completion time
- Setup: 1hr
- Running experiments: 10 hr

