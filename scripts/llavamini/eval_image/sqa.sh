#!/bin/bash
LLAVA_MINI_ROOT=.
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

model_path='./checkpoints/recasprefusion_mean4rec_withcond_lr3e-5_4l_all665k'
CKPT=all665k
echo "Model path is set to: $model_path"

DATA_ROOT=$LLAVA_MINI_ROOT/playground/data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $LLAVA_MINI_ROOT/llavamini/eval/model_vqa_science.py \
        --model-path ${model_path} \
        --question-file $DATA_ROOT/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder $DATA_ROOT/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_llama_3_1 --model-name llava-mini &
done

wait

output_file=./playground/data/eval/scienceqa/answers/$CKPT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llavamini/eval/eval_science_qa.py \
    --base-dir $DATA_ROOT/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$CKPT/merge_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$CKPT/merge_result.jsonl