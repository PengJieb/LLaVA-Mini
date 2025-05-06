#export HOME=/data3/tianlong
#  GIT_SSH_COMMAND="ssh -i  /data3/tianlong/.ssh/id_ed25519" git push -u origin dev

deepspeed --include localhost:4,5,6,7 --master_port 15354 llavamini/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version llava_llama_3_1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --compressor_size 1 --resolution_ratio 1 \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/in_prefusion_trainwithcompproj_alldata \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --max_grad_norm 0.5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --recurrent_in_compression False \
    --recurrent_in_prefusion True \
    --only_recurrent_trainable False \
    --recurrent_in_llm_residue False \
    --recurrent_in_llm_range 6 \
    --recurrent_in_prefusion_residue False \
    --n_layers_in_recurrent_block 1 \
    --freeze_backbone True \
    --report_to none

