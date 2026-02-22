from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import fire
import os
import torch
import sys
import json
import random

# Force GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_test_data(test_file):
    test_prompts = []
    target_responses = []
    with open(test_file, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            test_prompts.append(data["instruction"])
            target_responses.append(data["output"])
    return test_prompts, target_responses

def robust_evaluation(predict_responses, target_responses, label_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)
    
    # Normalize labels in dict to uppercase for consistency
    norm_label_dict = {k.upper(): v for k, v in label_dict.items()}
    sorted_labels = sorted(norm_label_dict.keys(), key=len, reverse=True) # Check longer labels first (e.g., 'WEB ATTACK' before 'WEB')

    preds = []
    labels = []
    mistakes = []

    for pred_raw, target_raw in zip(predict_responses, target_responses):
        pred_text = str(pred_raw).strip().upper()
        target_text = str(target_raw).strip().upper()

        # 1. Match prediction
        found_pred = None
        for label_key in sorted_labels:
            if label_key in pred_text:
                found_pred = norm_label_dict[label_key]
                break
        
        if found_pred is not None:
            preds.append(found_pred)
        else:
            # Mark as an 'Unknown' class index (max_labels)
            preds.append(len(norm_label_dict))
            mistakes.append(pred_raw)

        # 2. Match target
        if target_text in norm_label_dict:
            labels.append(norm_label_dict[target_text])
        else:
            # If target is not in dict, maybe it's nested or needs keyword match too
            found_target = None
            for label_key in sorted_labels:
                if label_key in target_text:
                    found_target = norm_label_dict[label_key]
                    break
            if found_target is not None:
                labels.append(found_target)
            else:
                print(f"Warning: Target label '{target_raw}' not found in label dictionary.")
                labels.append(len(norm_label_dict))

    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    print("\nClassification Report:")
    target_names = list(label_dict.keys())
    if len(set(preds + labels)) > len(target_names):
        target_names.append("MISTAKE/UNKNOWN")
    
    # Adjust target names to match used labels
    used_labels = sorted(list(set(preds + labels)))
    display_names = [target_names[i] if i < len(target_names) else "UNKNOWN" for i in used_labels]
    
    print(classification_report(labels, preds, target_names=display_names, labels=used_labels, zero_division=0))
    
    if mistakes:
        print(f"\nTotal Mistake Labels generated: {len(mistakes)}")
        print("Sample mistakes (first 5):", mistakes[:5])

def main(model_name,
         test_file: str,
         label_file: str,
         ptuning_path: str = None,
         num_samples: int = 2000,
         **kwargs):

    print(f"Loading Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if ptuning_path is not None:
        print(f"Loading P-Tuning weights from: {ptuning_path}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        # Load prefix weights
        prefix_state_dict = torch.load(os.path.join(ptuning_path, "pytorch_model.bin"), map_location="cpu")
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        model = model.half().cuda()
        model.transformer.prefix_encoder.float()
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

    model = model.eval()

    print(f"Loading Test Data: {test_file}")
    prompts, targets = load_test_data(test_file)
    
    if num_samples > 0:
        # Group by label to allow balanced sampling
        data_by_label = {}
        for p, t in zip(prompts, targets):
            label = str(t).strip().upper()
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append((p, t))
        
        sampled_prompts = []
        sampled_targets = []
        
        # Total samples will be num_samples (half per label)
        target_labels = ["BENIGN", "MALICIOUS"]
        per_label_n = num_samples // len(target_labels)
        
        random.seed(42)
        for label in target_labels:
            if label in data_by_label:
                samples = data_by_label[label]
                n = min(len(samples), per_label_n)
                picked = random.sample(samples, n)
                for p, t in picked:
                    sampled_prompts.append(p)
                    sampled_targets.append(t)
                print(f"Sampled {n} samples for label: {label}")
            else:
                print(f"Warning: Label '{label}' not found in test data.")
        
        # If the labels aren't strictly BENIGN/MALICIOUS (e.g. multi-class), 
        # fall back to original behavior or show warning.
        if not sampled_prompts:
            print("No BENIGN/MALICIOUS labels found, falling back to sequential sampling.")
            prompts = prompts[:num_samples]
            targets = targets[:num_samples]
        else:
            prompts = sampled_prompts
            targets = sampled_targets

    predict_responses = []

    print(f"Starting Inference on {len(prompts)} samples...")
    for i, prompt in enumerate(tqdm(prompts)):
        with torch.no_grad():
            # Use deterministic settings
            response, _ = model.chat(tokenizer, prompt, history=[], top_p=0.85, temperature=0.1)
        predict_responses.append(response)
        
        # Display each inference result
        print(f"\n[{i+1}/{len(prompts)}] Sample Detail:")
        print(f"  Instruction: {prompt[:100]}...")
        print(f"  Target:      {targets[i]}")
        print(f"  Prediction:  {response}")
        print("-" * 20)

    robust_evaluation(predict_responses, targets, label_file)

if __name__ == "__main__":
    fire.Fire(main)
