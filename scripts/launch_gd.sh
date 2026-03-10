# GD Training on SD15
MODEL_NAME="Lykon/DreamShaper"
n_gpus=8
acc_steps=1
num_updates=8000
accelerate launch --num_processes=$n_gpus --mixed_precision="bf16" train_sd15_gd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=$acc_steps \
  --max_train_steps=$(($num_updates*$acc_steps)) \
  --checkpointing_steps=$(($num_updates*$acc_steps/10)) \
  --n_gpus=$n_gpus \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0;


# GD Training on SDXL
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
n_gpus=8
acc_steps=1
num_updates=4000
accelerate launch --num_processes=$n_gpus --mixed_precision="bf16" train_sdxl_gd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=$acc_steps \
  --max_train_steps=$(($num_updates*$acc_steps)) \
  --checkpointing_steps=$(($num_updates*$acc_steps/10)) \
  --n_gpus=$n_gpus \
  --train_batch_size=8 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0;


# Baseline Sampling and evaluation
n_gpus=8
model_name="sd15"
steps=20
cfg_scale=5.0
batch_gpu=16
out_folder="${model_name}_base_steps${steps}_cfg${cfg_scale}"
image_path="./samples/${out_folder}"
desc="${out_folder}"
ref="/Path/to/reference/statistis"
accelerate launch --num_processes=$n_gpus sample.py --model_name=$model_name --steps=$steps --cfg_scale=$cfg_scale --batch_gpu=$batch_gpu;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_fid.py calc --images=$image_path --ref=$ref --desc=$desc;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_clip_score.py calc --images=$image_path --desc=$desc;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_aes_score.py calc --images=$image_path --desc=$desc;


# GD Sampling and evaluation
n_gpus=8
model_name="sd15"
model_path=0
model_id=$model_path
method="gd"
steps=20
gd_scale=5.0
cfg_scale=1.0
batch_gpu=16
out_folder="${model_name}_${method}_${model_id}_steps${steps}_cfg${cfg_scale}_gdScale${gd_scale}"
image_path="./samples/${out_folder}"
desc="${out_folder}"
ref="/Path/to/reference/statistis"
accelerate launch --num_processes=$n_gpus sample.py --model_name=$model_name --model_path=$model_path --model_id=$model_id --method=$method --steps=$steps --gd_scale=$gd_scale --cfg_scale=$cfg_scale --batch_gpu=$batch_gpu;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_fid.py calc --images=$image_path --ref=$ref --desc=$desc;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_clip_score.py calc --images=$image_path --desc=$desc;
accelerate launch --num_processes=$n_gpus ./evaluation/eval_aes_score.py calc --images=$image_path --desc=$desc;
