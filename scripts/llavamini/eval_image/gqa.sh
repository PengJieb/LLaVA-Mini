#!/bin/bash
LLAVA_MINI_ROOT='./'
# gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# model_path=./checkpoints/newriniteplaceprefusion_4l_sub66.5k/checkpoint-1
model_path=./checkpoints/recasprefusion_mean4rec_nocond_lr3e-5_4l_all665k/checkpoint-3500
# model_path=ICTNLP/llava-mini-llama-3.1-8b
#CKPT=$(basename "$CKPT")
echo "Model path is set to: $model_path"

SPLIT="llava_gqa_testdev_balanced"
GQA_DATA=$LLAVA_MINI_ROOT/playground/data
GQADIR=$LLAVA_MINI_ROOT/playground/data/eval/gqa/data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $LLAVA_MINI_ROOT/llavamini/eval/model_vqa_loader.py \
        --model-path ${model_path} \
        --question-file $GQA_DATA/eval/gqa/$SPLIT.jsonl \
        --image-folder $GQA_DATA/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_llama_3_1 --model-name llava-mini &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQA_DATA/eval/gqa/data/testdev_balanced_predictions.json

cd $GQA_DATA/eval/gqa/data
#python eval.py --tier testdev_balanced > ./res_recurrent_llmresidule_32.txt
python eval.py --tier testdev_balanced > ./res_recasprefusion_4rec_nocond_retrain_alldata_lr3e-5_8.txt
# python eval.py --tier testdev_balanced > ./res_new4linitreplaceprefusion_mean32_4.txt
# python eval.py --tier testdev_balanced > ./debug.txt
# python eval.py --tier testdev_balanced > ./official.txt

#cat ./res_recurrent_llmresidule_32.txt
cat ./res_recasprefusion_4rec_nocond_retrain_alldata_lr3e-5_8.txt
# cat ./res_new4linitreplaceprefusion_mean32_4.txt
# cat ./debug.txt
# cat ./official.txt
