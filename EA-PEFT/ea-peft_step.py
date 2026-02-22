import fire
import os

# Robust path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def get_abs_path(path):
    """Returns absolute path. If path is relative, it is resolved against the current working directory."""
    if not path:
        return None
    return os.path.abspath(path)


def stage1_tuning(model_name, instruction_data, validation_data=None, resume=False):
    tuner_script = os.path.join(PROJECT_ROOT, "dual-stage-tuning", "main.py")
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    output_dir = os.path.join(PROJECT_ROOT, "models", "chatglm2", "peft", "instruction")

    if validation_data is None:
        validation_data = instruction_data

    cmd = f"torchrun --standalone --nnodes=1 --nproc-per-node=1 {tuner_script} \
    --do_train \
    --train_file {instruction_data} \
    --validation_file {validation_data} \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir {cache_dir} \
    --model_name_or_path {model_name} \
    --output_dir {output_dir} \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500"

    if resume:
        cmd += " --resume_from_checkpoint True"

    os.system(cmd)


def stage2_tuning(model_name, traffic_data, task_name, validation_data=None, resume=False):
    tuner_script = os.path.join(PROJECT_ROOT, "dual-stage-tuning", "main.py")
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    output_dir = os.path.join(PROJECT_ROOT, "models", "chatglm2", "peft", task_name)

    if validation_data is None:
        validation_data = traffic_data

    cmd = f"torchrun --standalone --nnodes=1 --nproc-per-node=1 {tuner_script} \
    --do_train \
    --train_file {traffic_data} \
    --validation_file {validation_data} \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir {cache_dir} \
    --model_name_or_path {model_name} \
    --output_dir {output_dir} \
    --overwrite_output_dir \
    --max_source_length 1024\
    --max_target_length 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8\
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 20 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --pre_seq_len 128 \
    --warmup_steps 1500 \
    --bf16 True \
    --lr_scheduler_type cosine \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 1000"

    if resume:
        cmd += " --resume_from_checkpoint True"

    os.system(cmd)


def model_update(model_name, traffic_data, task_name, validation_data=None, resume=False):
    peft_dir = os.path.join(PROJECT_ROOT, "models", "chatglm2", "peft")
    if not os.path.exists(peft_dir):
        os.makedirs(peft_dir, exist_ok=True)
    
    if task_name in os.listdir(peft_dir):
         print(f"Warning: task {task_name} already exists in {peft_dir}")
    
    stage2_tuning(model_name, traffic_data, task_name, validation_data=validation_data, resume=resume)


def model_insert(model_name, traffic_data, task_name, validation_data=None, resume=False):
    peft_dir = os.path.join(PROJECT_ROOT, "models", "chatglm2", "peft")
    if not os.path.exists(peft_dir):
        os.makedirs(peft_dir, exist_ok=True)
    
    if task_name not in os.listdir(peft_dir):
        os.mkdir(os.path.join(peft_dir, task_name))
    
    stage2_tuning(model_name, traffic_data, task_name, validation_data=validation_data, resume=resume)


def main(model_name,
         tuning_data: str = None,
         adaptation_task: str = "update",
         task_name: str = "IDS",
         skip_stage1: bool = True,
         resume: bool = False,
         **kwargs):
    
    print(f"Starting EA-PEFT process...")
    print(f"  Model: {model_name}")
    print(f"  Data: {tuning_data}")
    print(f"  Task: {task_name} ({adaptation_task})")

    # Resolve input paths
    model_name = get_abs_path(model_name)
    tuning_data = get_abs_path(tuning_data)
    
    # Path setup
    instruction_path = os.path.join(tuning_data, "instructions/instruction.json")
    traffic_path = os.path.join(tuning_data, "traffic/traffic.json")
    val_path = None
    
    # Special case for IDS data
    if not os.path.exists(traffic_path):
        for prefix in ["binary_", "multi_"]:
            p_train = os.path.join(tuning_data, f"cicids_{prefix}train.json")
            p_val = os.path.join(tuning_data, f"cicids_{prefix}val.json")
            if os.path.exists(p_train):
                traffic_path = p_train
                if os.path.exists(p_val):
                    val_path = p_val
                break

    if not skip_stage1:
        if os.path.exists(instruction_path):
            print(f"Starting Stage 1 Tuning (Instruction)...")
            stage1_tuning(model_name, instruction_path, resume=resume)
        else:
            print(f"Warning: Instruction file not found at {instruction_path}, skipping Stage 1.")

    if os.path.exists(traffic_path):
        print(f"Starting Stage 2 Tuning (Traffic: {traffic_path})...")
        if val_path:
            print(f"  Validating with: {val_path}")
        
        if adaptation_task == "update":
            model_update(model_name, traffic_path, task_name, validation_data=val_path, resume=resume)
        elif adaptation_task == "register":
            model_insert(model_name, traffic_path, task_name, validation_data=val_path, resume=resume)
    else:
        print(f"Error: Traffic training file not found at {tuning_data}")


if __name__ == "__main__":
    fire.Fire(main)
