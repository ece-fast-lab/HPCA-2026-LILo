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
0. Prepare model parameters and software environment: [environment setup](./scripts/README.md)
1. Inference throughput and latency on DeepSeek-R1, Llama3-405B, Qwen3-235B, and OPT-175B ([Fig.11, Fig.13](./scripts/fig11.sh))
2. Energy effciency (Fig.14)

## Estimated completion time
Pretty long
