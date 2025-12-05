import os
import re

def extract_latencies(root_folder):
    latencies = []

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
                            latencies.append((file_path, value))
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    return latencies

if __name__ == "__main__":
    base_dir = "../scripts/step_1_baseline/results/"
    decmp_dir = "../scripts/step_2_decomp/results/"

    baseline_results = extract_latencies(base_dir)
    decmp_results = extract_latencies(decmp_dir)

    for file_path, val in baseline_results:
        print(f"{file_path}: {val} ms")

    print("\nTotal latencies found:", len(baseline_results))
