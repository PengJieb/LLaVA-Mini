#!/bin/bash
LLAVA_MINI_ROOT=.
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

model_path='./checkpoints/recasprefusion_mean4rec_withcond_lr3e-5_4l_all665k'
CKPT=all665k
echo "Model path is set to: $model_path"

SPLIT="llava_gqa_testdev_balanced"
DATA_ROOT=$LLAVA_MINI_ROOT/playground/data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $LLAVA_MINI_ROOT/llavamini/eval/model_vqa_loader.py \
        --model-path ${model_path} \
        --question-file $DATA_ROOT/eval/pope/llava_pope_test.jsonl \
        --image-folder $DATA_ROOT/eval/pope/val2014  \
        --answers-file ./playground/data/eval/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_llama_3_1 --model-name llava-mini &
done

wait

output_file=./playground/data/eval/pope/answers/$CKPT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_pope.py \
    --annotation-dir $DATA_ROOT/eval/pope/coco/annos \
    --question-file $DATA_ROOT/eval/pope/llava_pope_test.jsonl \
    --result-file $output_file > ./playground/data/eval/pope/answers/$CKPT/res_alldata_8.txt
cat ./playground/data/eval/pope/answers/$CKPT/res_alldata_8.txt