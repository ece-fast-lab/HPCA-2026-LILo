import os
import re
import matplotlib.pyplot as plt


def get_storage_tput():
    return 3.7 # GB/s

def get_llama_tput( 
    time, 
    is_comp,
    storage_tput, 
    mem_cap, 
    bs, 
    category
):
    LLAMA_SCALE = 6
    UNCOMP_FOOTPRINT = 754 #GB
    COMP_FOOTPRINT = UNCOMP_FOOTPRINT * 0.7086
    d_model = 16384
    n_layer = 126

    if category == 1:
        Lin = 128
        Lout = 256
    elif category == 2:
        Lin = 512
        Lout = 512
    elif category == 3:
        Lin = 1024
        Lout = 128
    elif category == 4:
        Lin = 1566
        Lout = 256
    else:
        print("Error: choose a valid category (between 1,2,3,4).")

    decomp_full_scale = time * LLAMA_SCALE
    if is_comp:
        weight_footprint = COMP_FOOTPRINT
    else:
        weight_footprint = UNCOMP_FOOTPRINT
    memory_footprint = weight_footprint + (2 * 2 * bs * (Lin+Lout) * d_model / 16 * n_layer \
                        + 2 * bs * (Lin + Lout) * d_model) / 1024 / 1024 / 1024
    storage_overhead = max(0, (memory_footprint-mem_cap)/storage_tput*Lout)
    return Lout*bs/(storage_overhead + decomp_full_scale) # throughput token/s

# print(get_llama_tput(340.724, True, 3.7, 512, 128, 256, 1))
# print(get_llama_tput(117.569, False, 3.7, 512, 128, 256, 1))

def get_ds_tput( 
    time, 
    is_comp,
    storage_tput, 
    mem_cap, 
    bs,
    category
):
    DS_SCALE = 3
    UNCOMP_FOOTPRINT = 1250 #GB
    COMP_FOOTPRINT = UNCOMP_FOOTPRINT * 0.683
    d_hidden = 7168
    d_c = 512
    n_layer = 61

    if category == 1:
        Lin = 128
        Lout = 256
    elif category == 2:
        Lin = 512
        Lout = 512
    elif category == 3:
        Lin = 1024
        Lout = 128
    elif category == 4:
        Lin = 1566
        Lout = 256
    else:
        print("Error: choose a valid category (between 1,2,3,4).")

    decomp_full_scale = time * DS_SCALE
    if is_comp:
        weight_footprint = COMP_FOOTPRINT
    else:
        weight_footprint = UNCOMP_FOOTPRINT
        
    memory_footprint = weight_footprint + (2 * 2 * bs * (Lin+Lout) * d_c * n_layer \
                        + 2 * bs * (Lin + Lout) * d_hidden) / 1024 / 1024 / 1024
    
    if category == 1:
        storage_overhead = max(0, (memory_footprint - mem_cap)/storage_tput*Lout*8/256/0.54)
    elif category == 2:
        storage_overhead = max(0, (memory_footprint - mem_cap)/storage_tput*Lout*30/256/0.82)
    elif category == 3:
        storage_overhead = max(0, (memory_footprint - mem_cap)/storage_tput*Lout*102/256/0.94)
    elif category == 4:
        storage_overhead = max(0, (memory_footprint - mem_cap)/storage_tput*Lout*223/256/0.97)
    else:
        print("Error: choose a valid category (between 1,2,3,4).")

    return Lout*bs/(storage_overhead + decomp_full_scale) # throughput token/s


def extract_latency(root_folder):
    latencies = {'llama':{}, 'ds':{}}

    # Regex pattern for: Inference latency: <float> ms
    pattern = re.compile(r"Inference latency:\s*([0-9]*\.?[0-9]+)\s*ms")

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        match = pattern.match(line)
                        if match:
                            value = float(match.group(1))
                            if "llama" in file_path:
                                latencies['llama'][fname] = value
                            elif "ds" in file_path:
                                latencies['ds'][fname] = value
                            # latencies[file_path] = value
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    return latencies

def sort_key(fname):
    m = re.match(r"c(\d+)-b(\d+)\.log", fname)
    c = int(m.group(1))
    b = int(m.group(2))
    return (c,b)

if __name__ == "__main__":
    base_dir = "../scripts/step_1_baseline/results/"
    decmp_dir = "../scripts/step_2_decomp/results/"

    storage_tput = get_storage_tput()

    baseline_results = extract_latency(base_dir)
    decmp_results = extract_latency(decmp_dir)

    #------ 512GB ------#
    mem_cap = 512 #GB

    llama_base = baseline_results['llama']
    llama_base_keys = sorted(llama_base.keys(), key=sort_key)
    llama_base_bs = [sort_key(llama_base_key)[1] for llama_base_key in llama_base_keys]
    llama_base_cat = [sort_key(llama_base_key)[0] for llama_base_key in llama_base_keys]
    llama_base_values = [llama_base[k] for k in llama_base_keys]
    llama_base_tput = [get_ds_tput(llama_base_values[idx], False, storage_tput, mem_cap, llama_base_bs[idx],llama_base_cat[idx] ) for idx in range(len(llama_base_values))]

    ds_decomp = decmp_results['llama']
    ds_decomp_keys = sorted(ds_decomp.keys(), key=sort_key)
    ds_decomp_bs = [sort_key(ds_decomp_key)[1] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_cat = [sort_key(ds_decomp_key)[0] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_values = [ds_decomp[k] for k in ds_decomp_keys]
    ds_decomp_tput = [get_ds_tput(ds_decomp_values[idx], True, storage_tput, mem_cap, ds_decomp_bs[idx],ds_decomp_cat[idx] ) for idx in range(len(ds_decomp_values))]
    #-------------------#

    #------- 1TB -------#
    mem_cap = 1024 #GB

    ds_base = baseline_results['ds']
    ds_base_keys = sorted(ds_base.keys(), key=sort_key)
    ds_base_bs = [sort_key(ds_base_key)[1] for ds_base_key in ds_base_keys]
    ds_base_cat = [sort_key(ds_base_key)[0] for ds_base_key in ds_base_keys]
    ds_base_values = [ds_base[k] for k in ds_base_keys]
    ds_base_tput = [get_ds_tput(ds_base_values[idx], False, storage_tput, mem_cap, ds_base_bs[idx],ds_base_cat[idx] ) for idx in range(len(ds_base_values))]

    ds_decomp = decmp_results['ds']
    ds_decomp_keys = sorted(ds_decomp.keys(), key=sort_key)
    ds_decomp_bs = [sort_key(ds_decomp_key)[1] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_cat = [sort_key(ds_decomp_key)[0] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_values = [ds_decomp[k] for k in ds_decomp_keys]
    ds_decomp_tput = [get_ds_tput(ds_decomp_values[idx], True, storage_tput, mem_cap, ds_decomp_bs[idx],ds_decomp_cat[idx] ) for idx in range(len(ds_decomp_values))]
    #-------------------#
    assert(len(ds_base_tput) == len(ds_decomp_tput))
    combined = [x for pair in zip(ds_base_tput, ds_decomp_tput) for x in pair]
    
