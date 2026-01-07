# python inference_MTD_2.py   --config config.json   --prompts prompts.jsonl   --out /home/rtx-5090/ching/pcap/malicious/mtd_results.csv   --device cuda:0   --temperature 0.1 --top_p 0.9 --max_new_tokens 32# -*- coding: utf-8 -*-
import os, json, csv, re, argparse, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ---- 你的類別集合（沿用你程式）----
BENIGN_APPS = {
    "BitTorrent","FTP","Facetime","Gmail","MySQL","Outlook","SMB","Skype","Weibo","WorldOfWarcraft"
}
MALWARE_FAMILIES = {
    "Cridex","Geodo","Htbot","Miuref","Neris","Nsis-ay","Shifu","Tinba","Virut","Zeus"
}
ALL_MTD_LABELS = BENIGN_APPS | MALWARE_FAMILIES

def normalize_mtd_label(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()
    for lab in ALL_MTD_LABELS:
        if low == lab.lower():
            return lab
    for lab in ALL_MTD_LABELS:
        if re.search(rf"\b{re.escape(lab.lower())}\b", low):
            return lab
    # 常見保底
    if re.search(r"\bbenign|normal\b", low): return "benign"
    if re.search(r"\bmalicious|malware\b", low): return "malicious"
    m = re.search(r"[A-Za-z][A-Za-z\-]*", t)
    return m.group(0) if m else t

def to_binary(label_or_text: str) -> str:
    s = (label_or_text or "").strip()
    if s in BENIGN_APPS: return "benign"
    if s in MALWARE_FAMILIES: return "malicious"
    low = s.lower()
    if any(lab.lower() in low for lab in BENIGN_APPS): return "benign"
    if any(lab.lower() in low for lab in MALWARE_FAMILIES): return "malicious"
    if low in {"benign","normal"}: return "benign"
    if low in {"malicious","malware"}: return "malicious"
    return "malicious"   # 保守

# ---- 和你現有 preprompt 對齊，但加上「只輸出一個標籤」限制 ----
def strict_mtd_preprompt(traffic_packets_block: str) -> str:
    labels = list(BENIGN_APPS) + list(MALWARE_FAMILIES)
    labels_str = ", ".join(labels)
    return (
        "Given the following traffic data <packet> that contains protocol fields and traffic features.\n"
        "Please conduct the ENCRYPTED MALWARE DETECTION TASK (MTD) to determine which application category the traffic belongs to.\n"
        f"Valid labels: {{{labels_str}}}.\n"
        "Output EXACTLY ONE label from the set above. No extra words.\n"
        + traffic_packets_block
    )

def load_model_with_prefix(model_path: str, peft_root: str, mtd_key: str, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True)

    # 載入 prefix-tuning
    prefix_state = torch.load(os.path.join(peft_root, mtd_key, "pytorch_model.bin"), map_location="cpu")
    new_prefix = {}
    for k, v in prefix_state.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix)

    if device.startswith("cuda"):
        model = model.half().to(device)
        model.transformer.prefix_encoder.float()
    else:
        model = model.to(device)
    model = model.eval()
    return tokenizer, model

def first_packets_block(text: str) -> str:
    # 你的 prompts.jsonl 的 text 已經從 <packet>: 開頭；保險起見還是截一下
    idx = text.find("<packet>:")
    return text[idx:] if idx != -1 else text

def run_batch(config_path: str, prompts_path: str, out_csv: str,
              device: str = "cuda:0", temperature: float = 0.1,
              top_p: float = 0.9, max_new_tokens: int = 32):

    # 讀設定
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_path = cfg["model_path"]
    peft_root  = cfg["peft_path"]
    mtd_key    = cfg["peft_set"]["MTD"]

    # 載模型（只載一次）
    tokenizer, model = load_model_with_prefix(model_path, peft_root, mtd_key, device=device)
    if hasattr(model, "generation_config"):
        model.generation_config.temperature    = temperature
        model.generation_config.top_p          = top_p
        model.generation_config.max_new_tokens = max_new_tokens

    total, pred_mal, unknown = 0, 0, 0
    t0 = time.time()

    with open(prompts_path, "r", encoding="utf-8") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["flow_id","n_packets","raw_response","normalized","binary"])

        for line in fin:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            flow_id   = rec.get("flow_id","")
            n_packets = rec.get("n_packets",0)
            text      = rec.get("text","")

            packets_block = first_packets_block(text)
            traffic_prompt = strict_mtd_preprompt(packets_block)

            with torch.no_grad():
                try:
                    resp, _ = model.chat(
                        tokenizer, traffic_prompt, history=[],
                        temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
                    )
                except TypeError:
                    resp, _ = model.chat(tokenizer, traffic_prompt, history=[])

            norm = normalize_mtd_label(resp)
            binary = to_binary(norm)

            total += 1
            if binary == "malicious":
                pred_mal += 1
            if norm not in ALL_MTD_LABELS and norm not in {"malicious","benign"}:
                unknown += 1

            writer.writerow([flow_id, n_packets, resp, norm, binary])

    dt = time.time() - t0
    ratio = pred_mal / max(total,1)
    print(f"✅ Done. {total} samples → malicious_pred={pred_mal} ({ratio:.2%}), unknown_norm={unknown}, time={dt:.1f}s")
    print(f"→ CSV saved to: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, help="你的 config.json 路徑")
    ap.add_argument("--prompts", required=True, help="prompts.jsonl（每行一筆，含 text）")
    ap.add_argument("--out",     default="mtd_results.csv", help="輸出 CSV")
    ap.add_argument("--device",  default="cuda:0", help="e.g., cuda:0 / cpu")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p",       type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    # 減少不必要開銷
    torch.set_grad_enabled(False)
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[1])

    run_batch(args.config, args.prompts, args.out,
              device=args.device,
              temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)

if __name__ == "__main__":
    main()
