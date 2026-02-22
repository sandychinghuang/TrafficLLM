import pandas as pd
import json
import os
import random
from tqdm import tqdm
import argparse

Train_Rate = 0.9
Val_Rate = 0.05

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="/home/user/ching/dataset/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling", help="Raw CSV directory")
    parser.add_argument("--output_path", type=str, default="/home/user/ching/TrafficLLM/done/CIC-IDS-2017", help="Output JSON path")
    parser.add_argument("--max_samples_per_class", type=int, default=100000, help="Max samples per category (set to 0 or less for ALL data)")
    parser.add_argument("--binary", action="store_true", help="Whether to perform binary classification (BENIGN vs MALICIOUS)")
    return parser.parse_args()

def preprocess_cicids_csv():
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    prefix = "binary_" if args.binary else "multi_"
    train_file = os.path.join(args.output_path, f"cicids_{prefix}train.json")
    val_file = os.path.join(args.output_path, f"cicids_{prefix}val.json")
    test_file = os.path.join(args.output_path, f"cicids_{prefix}test.json")

    csv_files = sorted([f for f in os.listdir(args.csv_dir) if f.endswith('.csv')])
    
    if args.binary:
        categories_str = "BENIGN, MALICIOUS"
    else:
        categories_str = "BENIGN, DDoS, PortScan, Bot, FTP-Patator, SSH-Patator, Web Attack, Infiltration, Heartbleed"
    
    instruction_base = f"Given the following network flow statistical features, please conduct the INTRUSION DETECTION TASK to determine if the traffic is {categories_str.replace(', ', ' or ')}. The categories include '{categories_str}'."

    # Pool for all collected data
    data_pool = {}

    for file in csv_files:
        file_path = os.path.join(args.csv_dir, file)
        print(f"Reading {file}...")
        
        try:
            chunks = pd.read_csv(file_path, chunksize=100000, low_memory=False)
            for chunk in chunks:
                chunk.columns = chunk.columns.str.strip()
                chunk['Label'] = chunk['Label'].astype(str).str.strip()
                
                if args.binary:
                    chunk['TargetLabel'] = chunk['Label'].apply(lambda x: 'BENIGN' if x.upper() == 'BENIGN' else 'MALICIOUS')
                else:
                    chunk['TargetLabel'] = chunk['Label']
                
                for label, group in chunk.groupby('TargetLabel'):
                    if label not in data_pool:
                        data_pool[label] = []
                    
                    # Convert to required format and add to pool
                    for _, row in group.iterrows():
                        features_dict = row.drop(['Label', 'TargetLabel']).to_dict()
                        feature_items = [f"{k}: {0 if pd.isnull(v) or v == float('inf') or v == float('-inf') else v}" 
                                       for k, v in features_dict.items()]
                        
                        feature_str = ", ".join(feature_items) or "All features are empty"
                        
                        data_pool[label].append({
                            "instruction": f"{instruction_base}\n<flow>: {feature_str}",
                            "output": str(label)
                        })
                del chunk
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print("\nData collection complete. Balancing and splitting...")

    final_train, final_val, final_test = [], [], []

    # If binary, ensure 1:1 balance
    if args.binary and "BENIGN" in data_pool and "MALICIOUS" in data_pool:
        min_count = min(len(data_pool["BENIGN"]), len(data_pool["MALICIOUS"]))
        if args.max_samples_per_class > 0:
            min_count = min(min_count, args.max_samples_per_class)
        
        print(f"Balancing: Truncating both BENIGN and MALICIOUS to {min_count} samples.")
        random.seed(42)
        for label in ["BENIGN", "MALICIOUS"]:
            samples = data_pool[label]
            random.shuffle(samples)
            samples = samples[:min_count]
            
            # Split 9:0.5:0.5
            n = len(samples)
            n_train = int(n * Train_Rate)
            n_val = int(n * Val_Rate)
            
            final_train.extend(samples[:n_train])
            final_val.extend(samples[n_train:n_train+n_val])
            final_test.extend(samples[n_train+n_val:])
    else:
        # For non-binary or single label, just split everything 8:1:1 per class
        for label, samples in data_pool.items():
            if args.max_samples_per_class > 0:
                samples = samples[:args.max_samples_per_class]
            
            random.seed(42)
            random.shuffle(samples)
            
            n = len(samples)
            n_train = int(n * Train_Rate)
            n_val = int(n * Val_Rate)
            
            print(f"  Label '{label}': total {n} samples -> {n_train}/{n_val}/{n - n_train - n_val}")
            
            final_train.extend(samples[:n_train])
            final_val.extend(samples[n_train:n_train+n_val])
            final_test.extend(samples[n_train+n_val:])

    # Final shuffle for each split
    random.shuffle(final_train)
    random.shuffle(final_val)
    random.shuffle(final_test)

    # Write to disk
    print(f"Writing results...")
    for file_path, data in [(train_file, final_train), (val_file, final_val), (test_file, final_test)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\nPreprocessing complete.")
    print(f"Total Train: {len(final_train)}")
    print(f"Total Val: {len(final_val)}")
    print(f"Total Test: {len(final_test)}")
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    preprocess_cicids_csv()
