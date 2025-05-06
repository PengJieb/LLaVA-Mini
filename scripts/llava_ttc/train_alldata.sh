#export HOME=/data3/tianlong
#  GIT_SSH_COMMAND="ssh -i  /data3/tianlong/.ssh/id_ed25519" git push -u origin dev

# --deepspeed ./scripts/zero2.json \

deepspeed --include localhost:4,5,6,7 --master_port 15354 llavamini/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ICTNLP/llava-mini-llama-3.1-8b \
    --version llava_llama_3_1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --compressor_size 1 --resolution_ratio 1 \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/recasprefusion_mean4rec_nocond_lr3e-5_4l_all665k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --max_grad_norm 1. \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --reinit_projector False \
    --recurrent_in_compression False \
    --recurrent_in_prefusion False \
    --recurrent_as_prefusion True \
    --recurrent_with_tcond False \
    --recurrent_as_llm False \
    --only_recurrent_trainable True \
    --recurrent_in_llm_residue False \
    --recurrent_in_llm_range 6 \
    --recurrent_in_prefusion_residue False \
    --n_layers_in_recurrent_block 4 \
    --mean_recurrence 4 \
    --mean_backprop_depth 1 \
    --recurrent_start_idx 28 \
    --freeze_backbone False \
    --report_to none

