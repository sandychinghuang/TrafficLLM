from transformers import AutoTokenizer, AutoModel, AutoConfig
import fire
import torch
import json
import os
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BENIGN_APPS = {
    "BitTorrent","FTP","Facetime","Gmail","MySQL","Outlook","SMB","Skype","Weibo","WorldOfWarcraft"
}
MALWARE_FAMILIES = {
    "Cridex","Geodo","Htbot","Miuref","Neris","Nsis-ay","Shifu","Tinba","Virut","Zeus"
}
ALL_MTD_LABELS = BENIGN_APPS | MALWARE_FAMILIES

def normalize_mtd_label(text: str) -> str:
    """
    把模型回覆的自由文字盡力對齊到 20 個 MTD 既定標籤之一。
    - 完全相等（忽略大小寫）
    - 或回覆句子中包含該標籤字樣
    對不上就原樣回傳（後續仍可用 to_binary 嘗試判定）。
    """
    t = (text or "").strip()
    low = t.lower()
    for lab in ALL_MTD_LABELS:
        if low == lab.lower():
            return lab
    for lab in ALL_MTD_LABELS:
        if re.search(rf"\b{re.escape(lab.lower())}\b", low):
            return lab
    return t


def to_binary(label_or_text: str) -> str:
    """
    多類 -> 二類（benign/malicious）。
    優先精準比對；其次子字串包含。
    若仍無法判定，為避免漏報，保守回傳 'malicious'。
    如需調整保守策略，可改成回傳 'unknown'。
    """
    s = (label_or_text or "").strip()
    if s in BENIGN_APPS:
        return "benign"
    if s in MALWARE_FAMILIES:
        return "malicious"
    low = s.lower()
    if any(lab.lower() in low for lab in BENIGN_APPS):
        return "benign"
    if any(lab.lower() in low for lab in MALWARE_FAMILIES):
        return "malicious"
    return "malicious"


def load_model(model, ptuning_path):
    if ptuning_path is not None:
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float()

    return model


def prompt_processing(prompt):
    instruction_text = prompt.split("<packet>")[0]
    traffic_data = "<packet>" + "<packet>".join(prompt.split("<packet>")[1:])

    return instruction_text, traffic_data


def preprompt(task, traffic_data):
    """Preprompts in LLMs for downstream traffic pattern learning"""
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application "
               "category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, "
               "FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, "
               "Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the "
               "traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
        "WAD": "Classify the given HTTP request into benign and malicious categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'malicious' or 'benign', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the encrypted VPN detection task to determine which behavior or "
               "application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, "
               "email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n",
        "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application "
               "category the traffic belongs to under the Tor network. The categories include 'audio, browsing, chat, "
               "file, mail, p2p, video, voip'.\n"
    }
    if task == "AAD":
        prompt = prepromt_set[task] + traffic_data.split("<packet>:")[1]
    else:
        prompt = prepromt_set[task] + traffic_data
    return prompt


def main(config, prompt: str = None, task: str = None, **kwargs):
    instruction_text, traffic_data = prompt_processing(prompt)

    with open(config, "r", encoding="utf-8") as fin:
        config = json.load(fin)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)
    
    if task is not None:
        ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
        model_downstream = load_model(model, ptuning_path).eval()
        traffic_prompt = preprompt(task, traffic_data)
        response, history = model_downstream.chat(tokenizer, traffic_prompt, history=[])
        print(response)
        
        if task == "MTD":
            binary = to_binary(normalize_mtd_label(response))
            print(f"[BINARY] {binary}")
        return
    
    # Stage 1: task understanding
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"]["NLP"])
    model_nlp = load_model(model, ptuning_path)

    model_nlp = model_nlp.eval()

    response, history = model_nlp.chat(tokenizer, instruction_text, history=[])
    print(response)

    # Stage 2: task-specific traffic learning
    task = config["tasks"][response]
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
    model_downstream = load_model(model, ptuning_path)

    model_downstream = model_downstream.eval()

    traffic_prompt = preprompt(task, traffic_data)
    response, history = model_downstream.chat(tokenizer, traffic_prompt, history=[])
    print(response)

    if task == "MTD":
        binary = to_binary(normalize_mtd_label(response))
        print(f"[BINARY] {binary}")


if __name__ == "__main__":
    fire.Fire(main)
