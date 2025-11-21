# Training DICE sharpener on SD15
MODEL_NAME="Lykon/DreamShaper"
n_gpus=8
acc_steps=1
num_updates=8000
accelerate launch --num_processes=$n_gpus --mixed_precision="fp16" train_sd15.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=$acc_steps \
  --max_train_steps=$(($num_updates*$acc_steps)) \
  --checkpointing_steps=$(($num_updates*$acc_steps/10)) \
  --n_gpus=$n_gpus \
  --guidance_scale=5 \
  --train_batch_size=16 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0;


# Training DICE sharpener on SDXL
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
n_gpus=8
acc_steps=2
num_updates=4000
accelerate launch --num_processes=$n_gpus --mixed_precision="bf16" train_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=$acc_steps \
  --max_train_steps=$(($num_updates*$acc_steps)) \
  --checkpointing_steps=$(($num_updates*$acc_steps/10)) \
  --n_gpus=$n_gpus \
  --guidance_scale=5 \
  --train_batch_size=8 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0;


# Sampling and evaluation with base models
model_name="sd15"
steps=20
cfg_scale=5.0
batch_gpu=16
out_folder="${model_name}_base_steps${steps}_cfg${cfg_scale}"
image_path="./samples/${out_folder}"
desc="${out_folder}"
ref="/Path/to/reference/statistis"
accelerate launch --num_processes=8 sample.py --model_name=$model_name --steps=$steps --cfg_scale=$cfg_scale --batch_gpu=$batch_gpu;
accelerate launch --num_processes=8 ./evaluation/eval_fid.py calc --images=$image_path --ref=$ref --desc=$desc;
accelerate launch --num_processes=8 ./evaluation/eval_clip_score.py calc --images=$image_path --desc=$desc;
accelerate launch --num_processes=8 ./evaluation/eval_aes_score.py calc --images=$image_path --desc=$desc;


# Sampling and evaluation using trained DICE sharpener
model_name="sd15"
sharpener_path=0
sharpener_id=$sharpener_path
steps=20
alpha=1.0
batch_gpu=16
out_folder="${model_name}_${sharpener_id}_steps${steps}_alpha${alpha}"
image_path="./samples/${out_folder}"
desc="${out_folder}"
ref="/Path/to/reference/statistis"
accelerate launch --num_processes=8 sample.py --model_name=$model_name --sharpener_path=$sharpener_path --sharpener_id=$sharpener_id --steps=$steps --alpha=$alpha --batch_gpu=$batch_gpu;
accelerate launch --num_processes=8 ./evaluation/eval_fid.py calc --images=$image_path --ref=$ref --desc=$desc;
accelerate launch --num_processes=8 ./evaluation/eval_clip_score.py calc --images=$image_path --desc=$desc;
accelerate launch --num_processes=8 ./evaluation/eval_aes_score.py calc --images=$image_path --desc=$desc;

