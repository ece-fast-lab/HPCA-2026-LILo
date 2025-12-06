import os
import re
import matplotlib.pyplot as plt
import numpy as np


def get_storage_tput():
    return 3.7 # GB/s

def get_llama_latencies(
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

    time_full_scale = time * LLAMA_SCALE
    if is_comp:
        weight_footprint = COMP_FOOTPRINT
    else:
        weight_footprint = UNCOMP_FOOTPRINT
    memory_footprint = weight_footprint + (2 * 2 * bs * (Lin+Lout) * d_model / 16 * n_layer \
                        + 2 * bs * (Lin + Lout) * d_model) / 1024 / 1024 / 1024
    storage_overhead = max(0, (memory_footprint-mem_cap)/storage_tput*Lout)
    return storage_overhead, time_full_scale

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

def get_ds_latencies(
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

    return storage_overhead, decomp_full_scale # throughput token/s


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

def get_subplot1_data(baseline_results, decmp_results, storage_tput):
    # pass
    #------ 512GB ------#
    mem_cap = 512 #GB

    llama_base = baseline_results['llama']
    llama_base_keys = sorted(llama_base.keys(), key=sort_key)
    llama_base_bs = [sort_key(llama_base_key)[1] for llama_base_key in llama_base_keys]
    llama_base_cat = [sort_key(llama_base_key)[0] for llama_base_key in llama_base_keys]
    llama_base_values = [llama_base[k] for k in llama_base_keys]
    llama_base_tput = [get_llama_tput(llama_base_values[idx], False, storage_tput, mem_cap, llama_base_bs[idx],llama_base_cat[idx]) for idx in range(len(llama_base_values))]

    llama_decomp = decmp_results['llama']
    llama_decomp_keys = sorted(llama_decomp.keys(), key=sort_key)
    llama_decomp_bs = [sort_key(llama_decomp_key)[1] for llama_decomp_key in llama_decomp_keys]
    llama_decomp_cat = [sort_key(llama_decomp_key)[0] for llama_decomp_key in llama_decomp_keys]
    llama_decomp_values = [llama_decomp[k] for k in llama_decomp_keys]
    llama_decomp_tput = [get_llama_tput(llama_decomp_values[idx], True, storage_tput, mem_cap, llama_decomp_bs[idx],llama_decomp_cat[idx]) for idx in range(len(llama_decomp_values))]
    
    assert(len(llama_base_tput) == len(llama_decomp_tput))
    llama_combined = [x for pair in zip(llama_base_tput, llama_decomp_tput) for x in pair]
    # print(llama_combined)
    #-------------------#

    #------- 1TB -------#
    mem_cap = 1024 #GB

    ds_base = baseline_results['ds']
    ds_base_keys = sorted(ds_base.keys(), key=sort_key)
    ds_base_bs = [sort_key(ds_base_key)[1] for ds_base_key in ds_base_keys]
    ds_base_cat = [sort_key(ds_base_key)[0] for ds_base_key in ds_base_keys]
    ds_base_values = [ds_base[k] for k in ds_base_keys]
    ds_base_tput = [get_ds_tput(ds_base_values[idx], False, storage_tput, mem_cap, ds_base_bs[idx],ds_base_cat[idx]) for idx in range(len(ds_base_values))]

    ds_decomp = decmp_results['ds']
    ds_decomp_keys = sorted(ds_decomp.keys(), key=sort_key)
    ds_decomp_bs = [sort_key(ds_decomp_key)[1] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_cat = [sort_key(ds_decomp_key)[0] for ds_decomp_key in ds_decomp_keys]
    ds_decomp_values = [ds_decomp[k] for k in ds_decomp_keys]
    ds_decomp_tput = [get_ds_tput(ds_decomp_values[idx], True, storage_tput, mem_cap, ds_decomp_bs[idx],ds_decomp_cat[idx]) for idx in range(len(ds_decomp_values))]

    assert(len(ds_base_tput) == len(ds_decomp_tput))
    ds_combined = [x for pair in zip(ds_base_tput, ds_decomp_tput) for x in pair]
    # print(ds_combined)
    #-------------------#
    return llama_base_tput, llama_decomp_tput, llama_decomp_keys, ds_base_tput, ds_decomp_tput, ds_decomp_keys

def get_subplot2_data(baseline_results, decmp_results, storage_tput):
    # pass
    # baseline_results
    llama_data  = {'lilo_compute':[], 'lilo_decomp':[], 'lilo_storage':[], 'base_compute':[], 'base_storage':[]}
    ds_data     = {'lilo_compute':[], 'lilo_decomp':[], 'lilo_storage':[], 'base_compute':[], 'base_storage':[]}
    #------ 512GB ------#
    mem_cap = 512 #GB
    llama_base = baseline_results['llama']
    llama_decomp = decmp_results['llama']

    llama_base_keys = sorted(llama_base.keys(), key=sort_key)
    filtered_base_keys = [key for key in llama_base_keys if "b4" not in key and "b16" not in key]
    llama_base_bs = [sort_key(llama_base_key)[1] for llama_base_key in filtered_base_keys]
    llama_base_cat = [sort_key(llama_base_key)[0] for llama_base_key in filtered_base_keys]
    llama_base_values = [llama_base[k] for k in filtered_base_keys]

    llama_decomp_keys = sorted(llama_decomp.keys(), key=sort_key)
    filtered_decomp_keys = [key for key in llama_decomp_keys if "b4" not in key and "b16" not in key]
    llama_decomp_bs = [sort_key(llama_decomp_key)[1] for llama_decomp_key in filtered_decomp_keys]
    llama_decomp_cat = [sort_key(llama_decomp_key)[0] for llama_decomp_key in filtered_decomp_keys]
    llama_decomp_values = [llama_decomp[k] for k in filtered_decomp_keys]

    llama_base_latencies = [
        get_llama_latencies(
            llama_base[k], 
            False, 
            storage_tput, 
            mem_cap, 
            sort_key(k)[1],
            sort_key(k)[0]
        ) for k in filtered_base_keys
    ] # return (storage_overhead, time_full_scale)
    llama_decomp_latencies = [
        get_llama_latencies(
            llama_decomp[k], 
            True, 
            storage_tput, 
            mem_cap, 
            sort_key(k)[1],
            sort_key(k)[0]
        ) for k in filtered_decomp_keys
    ] # return (storage_overhead, time_full_scale)
    llama_data['lilo_compute'] = [llama_base_latencies[idx][1] for idx in range(len(llama_base_latencies))]
    llama_data['lilo_storage'] = [llama_decomp_latencies[idx][0] for idx in range(len(llama_decomp_latencies))]
    llama_data['lilo_decomp'] = [llama_decomp_latencies[idx][1]-llama_base_latencies[idx][1] for idx in range(len(llama_decomp_latencies))]
    llama_data['base_compute'] = [llama_base_latencies[idx][1] for idx in range(len(llama_base_latencies))]
    llama_data['base_storage'] = [llama_base_latencies[idx][0] for idx in range(len(llama_base_latencies))]


    #------ 1TB ------#
    mem_cap = 1024 #GB
    ds_base = baseline_results['ds']
    ds_decomp = decmp_results['ds']


    ds_base_keys = sorted(ds_base.keys(), key=sort_key)
    filtered_base_keys = [key for key in ds_base_keys if "b4" not in key and "b16" not in key]
    ds_base_bs = [sort_key(ds_base_key)[1] for ds_base_key in filtered_base_keys]
    ds_base_cat = [sort_key(ds_base_key)[0] for ds_base_key in filtered_base_keys]
    ds_base_values = [ds_base[k] for k in filtered_base_keys]
    ds_decomp_keys = sorted(ds_decomp.keys(), key=sort_key)
    filtered_decomp_keys = [key for key in ds_decomp_keys if "b4" not in key and "b16" not in key]
    ds_decomp_bs = [sort_key(ds_decomp_key)[1] for ds_decomp_key in filtered_decomp_keys]
    ds_decomp_cat = [sort_key(ds_decomp_key)[0] for ds_decomp_key in filtered_decomp_keys]
    ds_decomp_values = [ds_decomp[k] for k in filtered_decomp_keys]

    ds_base_latencies = [
        get_ds_latencies(
            ds_base[k], 
            False, 
            storage_tput, 
            mem_cap, 
            sort_key(k)[1],
            sort_key(k)[0]
        ) for k in filtered_base_keys
    ] # return (storage_overhead, time_full_scale)
    ds_decomp_latencies = [
        get_ds_latencies(
            ds_decomp[k], 
            True, 
            storage_tput, 
            mem_cap, 
            sort_key(k)[1],
            sort_key(k)[0]
        ) for k in filtered_decomp_keys
    ] # return (storage_overhead, time_full_scale)
    ds_data['lilo_compute'] = [ds_base_latencies[idx][1] for idx in range(len(ds_base_latencies))]
    ds_data['lilo_storage'] = [ds_decomp_latencies[idx][0] for idx in range(len(ds_decomp_latencies))]
    ds_data['lilo_decomp'] = [ds_decomp_latencies[idx][1]-ds_base_latencies[idx][1] for idx in range(len(ds_decomp_latencies))]
    ds_data['base_compute'] = [ds_base_latencies[idx][1] for idx in range(len(ds_base_latencies))]
    ds_data['base_storage'] = [ds_base_latencies[idx][0] for idx in range(len(ds_base_latencies))]
    return llama_data, ds_data
    

if __name__ == "__main__":
    base_dir = "../scripts/step_1_baseline/results/"
    decmp_dir = "../scripts/step_2_decomp/results/"

    storage_tput = get_storage_tput()

    baseline_results = extract_latency(base_dir)
    decmp_results = extract_latency(decmp_dir)

    llama_base_tput, llama_decomp_tput, llama_decomp_keys, ds_base_tput, ds_decomp_tput, ds_decomp_keys = get_subplot1_data(baseline_results, decmp_results, storage_tput)
    llama_break, ds_break = get_subplot2_data(baseline_results, decmp_results, storage_tput)
    print(llama_break)
    print(ds_break)
    
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))

    indices = np.arange(len(llama_base_tput))
    bar_width = 0.4
    axes[0, 0].grid(True)
    axes[0, 0].bar(indices - bar_width/2, llama_base_tput, width=bar_width, label="Baseline")
    axes[0, 0].bar(indices + bar_width/2, llama_decomp_tput, width=bar_width, label="LILo")
    axes[0, 0].set_xticks(indices)
    axes[0, 0].set_xticklabels(llama_decomp_keys)
    axes[0, 0].set_xlabel("c<X>-b<Y> = Category X, Batch Size Y")
    axes[0, 0].set_ylabel("Throughput (tokens/s)")
    axes[0, 0].set_title("Llama3-405B, 512GB System Memory")

    indices = np.arange(len(ds_base_tput))
    bar_width = 0.4
    axes[0, 1].grid(True)
    axes[0, 1].bar(indices - bar_width/2, ds_base_tput, width=bar_width, label="Baseline")
    axes[0, 1].bar(indices + bar_width/2, ds_decomp_tput, width=bar_width, label="LILo")
    axes[0, 1].set_xticks(indices)
    axes[0, 1].set_xticklabels(ds_decomp_keys)
    axes[0, 1].set_xlabel("c<X>-b<Y> = Category X, Batch Size Y")
    axes[0, 1].set_ylabel("Throughput (tokens/s)")
    axes[0, 1].set_title("DeepSeek-R1, 1TB System Memory")

    ######## Subplot 2 ########
    num_dp = len(llama_break['lilo_compute'])
    dp_indices = np.arange(num_dp)
    base_norm = []
    lilo_norm = []

    for i in range(num_dp):
        compute = llama_break['base_compute'][i]
        storage = llama_break['base_storage'][i]
        base_total = compute + storage

        lilo_comp = llama_break['lilo_compute'][i] / base_total
        lilo_decomp = llama_break['lilo_decomp'][i] / base_total
        lilo_storage = llama_break['lilo_storage'][i] / base_total

        base_norm.append([compute / base_total, storage / base_total])
        lilo_norm.append([lilo_comp, lilo_decomp, lilo_storage])

    base_norm = np.array(base_norm)
    lilo_norm = np.array(lilo_norm)

    axes[1, 0].grid(True)
    axes[1, 0].bar(dp_indices + bar_width/2, base_norm[:,0], width = bar_width, label="Baseline-compute")
    axes[1, 0].bar(dp_indices + bar_width/2, base_norm[:,1], width = bar_width, bottom=base_norm[:,0], label="Baseline-storage")
    axes[1, 0].bar(dp_indices - bar_width/2, lilo_norm[:,0], width = bar_width, label="LILo-compute")
    axes[1, 0].bar(dp_indices - bar_width/2, lilo_norm[:,1], width = bar_width, bottom=lilo_norm[:,0], label="LILo-decompress")
    axes[1, 0].bar(dp_indices - bar_width/2, lilo_norm[:,2], width = bar_width, bottom=np.array(lilo_norm)[:,0]+np.array(lilo_norm)[:,1], label="LILo-storage")
    axes[1, 0].set_xticks(dp_indices)
    filtered_keys = [key for key in llama_decomp_keys if "b4" not in key and "b16" not in key]
    axes[1, 0].set_xticklabels(filtered_keys)
    axes[1, 0].set_xlabel("c<X>-b<Y> = Category X, Batch Size Y")
    axes[1, 0].set_ylabel("Normalized latency breakdown")
    axes[1, 0].set_title("Llama3-405B Latency Breakdown (Normalized)")
    axes[1, 0].legend()


    num_dp = len(ds_break['lilo_compute'])
    dp_indices = np.arange(num_dp)
    base_norm = []
    lilo_norm = []

    for i in range(num_dp):
        compute = ds_break['base_compute'][i]
        storage = ds_break['base_storage'][i]
        base_total = compute + storage

        lilo_comp = ds_break['lilo_compute'][i] / base_total
        lilo_decomp = ds_break['lilo_decomp'][i] / base_total
        lilo_storage = ds_break['lilo_storage'][i] / base_total

        base_norm.append([compute / base_total, storage / base_total])
        lilo_norm.append([lilo_comp, lilo_decomp, lilo_storage])

    base_norm = np.array(base_norm)
    lilo_norm = np.array(lilo_norm)

    axes[1, 1].grid(True)
    axes[1, 1].bar(dp_indices + bar_width/2, base_norm[:,0], width = bar_width, label="Baseline-compute")
    axes[1, 1].bar(dp_indices + bar_width/2, base_norm[:,1], width = bar_width, bottom=base_norm[:,0], label="Baseline-storage")
    axes[1, 1].bar(dp_indices - bar_width/2, lilo_norm[:,0], width = bar_width, label="LILo-compute")
    axes[1, 1].bar(dp_indices - bar_width/2, lilo_norm[:,1], width = bar_width, bottom=lilo_norm[:,0], label="LILo-decompress")
    axes[1, 1].bar(dp_indices - bar_width/2, lilo_norm[:,2], width = bar_width, bottom=np.array(lilo_norm)[:,0]+np.array(lilo_norm)[:,1], label="LILo-storage")
    axes[1, 1].set_xticks(dp_indices)
    filtered_keys = [key for key in llama_decomp_keys if "b4" not in key and "b16" not in key]
    axes[1, 1].set_xticklabels(filtered_keys)
    axes[1, 1].set_xlabel("c<X>-b<Y> = Category X, Batch Size Y")
    axes[1, 1].set_ylabel("Normalized latency breakdown")
    axes[1, 1].set_title("Llama3-405B Latency Breakdown (Normalized)")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig("fig11_llama.png")