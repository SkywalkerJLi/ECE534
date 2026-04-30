#!/bin/bash
#SBATCH --job-name=hf_act_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /home/sl5183/ECE534/.venv/bin/activate

python lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=nc8304/so101_combined \
  --dataset.root=/home/sl5183/.cache/huggingface/lerobot/nc8304/so101_combined \
  --policy.type=act \
  --output_dir=outputs/train/act_so101 \
  --job_name=act_so101\
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.repo_id=/scratch/gpfs/TSILVER/sl5183/ECE534 \
  --config_path=outputs/train/act_so101/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
