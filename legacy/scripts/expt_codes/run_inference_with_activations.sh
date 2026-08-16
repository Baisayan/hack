#!/bin/bash
#SBATCH --job-name=inf_w_activ
#SBATCH --gres=gpu:1
#SBATCH --partition=PA100q
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp


# === GPU Debug ===
echo "=== GPU Debug ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"
echo "================="

python run_inference_with_activations.py \
  --model_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/models/Qwen2.5-7B-Instruct \
  --model_tag qwen_7b_instruct \
  --input_csv_path /export/home2/sati0004/SouthAI-Safety-hackathon/XSTest.csv \
  --prompt_col prompt \
  --label_col label \
  --output_dir /export/home2/sati0004/SouthAI-Safety-hackathon/output/xstest \
  --max_new_tokens 256 \
  --dtype bfloat16 \
  --dataset mixed
