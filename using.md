### Env
```
conda create -n Traffic python=3.10 -y
conda activate Traffic

# Clone our TrafficLLM
git clone https://github.com/ZGC-LLM-Safety/TrafficLLM.git
cd TrafficLLM

git clone https://github.com/sandychinghuang/TrafficLLM.git

# Install required libraries
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install mkl==2021.4.0 intel-openmp==2021.4.0 mkl-service==2.4.0 -y
conda install scikit-learn scapy fire sentencepiece protobuf -c conda-forge -y
pip install transformers==4.30.2
pip install fire cpm_kernels mdtex2html accelerate sse-starlette flowcontainer

pip install kagglehub[pandas-datasets]

pip install "transformers>=4.36.0,<4.40.0" "huggingface-hub<1.0.0" "datasets"
pip install "transformers==4.30.2" "accelerate==0.21.0" "datasets==2.12.0" "huggingface-hub==0.16.4"
pip install "numpy==1.24.3" "sentencepiece" "cpm_kernels" "tokenizers==0.13.3" "protobuf==3.20.0"
pip install "fsspec==2023.6.0"

conda env export -n Traffic --no-builds > traffic_4090.yml
```

### Data Preprocess:
```
wsl
cd TrafficLLM
conda activate trafficllm

conda activate Traffic

python preprocess/preprocess_cicids.py --max_samples_per_class 500000 --binary
python preprocess/preprocess_cicids.py --max_samples_per_class 100000 --binary

python preprocess/preprocess_cicids.py --max_samples_per_class 0 --binary
```
### Tuning On Ada6000
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
rm -rf /home/user/ching/TrafficLLM/cache/*
rm -rf ~/.cache/huggingface/datasets/*
python ea-peft.py --model_name /home/user/ching/TrafficLLM/models/chatglm2/chatglm2-6b --tuning_data /home/user/ching/TrafficLLM/done/CIC-IDS-2017 --task_name IDS


python evaluation_cicids.py --model_name /home/user/ching/TrafficLLM/models/chatglm2/chatglm2-6b  --traffic_task detection --test_file /home/user/ching/TrafficLLM/done/CIC-IDS-2017/cicids_binary_test.json --label_file /home/user/ching/TrafficLLM/done/CIC-IDS-2017/label.json --ptuning_path /home/user/ching/TrafficLLM/models/chatglm2/peft/IDS/checkpoint-12000

```
### Tuning On RTX4090
```
cd ching/TrafficLLM/EA-PEFT

conda activate Traffic
conda install -c conda-forge nltk jieba
pip install rouge_chinese datasets

rm -rf /home/rtx4090/ching/TrafficLLM/cache/*
rm -rf ~/.cache/huggingface/datasets/*

python ea-peft.py --model_name /home/rtx4090/ching/TrafficLLM/models/chatglm2/chatglm2-6b --tuning_data /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017 --task_name IDS
python ea-peft.py --model_name /home/rtx4090/ching/TrafficLLM/models/chatglm2/chatglm2-6b --tuning_data /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017 --task_name IDS --resume True

cd ..
python evaluation_cicids.py --model_name /home/rtx4090/ching/TrafficLLM/models/chatglm2/chatglm2-6b  --traffic_task detection --test_file /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017/cicids_binary_test.json --label_file /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017/label.json --ptuning_path /home/rtx4090/ching/TrafficLLM/models/chatglm2/peft/IDS/checkpoint-12000
python evaluation_bak_2.py --model_name /home/rtx4090/ching/TrafficLLM/models/chatglm2/chatglm2-6b  --traffic_task detection --test_file /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017/cicids_binary_test.json --label_file /home/rtx4090/ching/TrafficLLM/done/CIC-IDS-2017/label.json --ptuning_path /home/rtx4090/ching/TrafficLLM/models/chatglm2/peft/IDS/checkpoint-10000
```