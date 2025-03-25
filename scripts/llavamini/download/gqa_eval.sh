MAXIMUM_CONNECTION_PER_SERVER=16
DOWNLOAD_CONNECTIONS=16
PROJ_ROOT=$(pwd)
echo $PROJ_ROOT


# echo "Download eval.zip, custom annotations, scripts, and the prediction files with LLaVA v1.5."
# mkdir playground/data/eval
# gdown 1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy -O playground/data/eval/eval.zip
# cd playground/data/eval
# unzip eval.zip
# rm eval.zip
# cd -

echo "Eval setup for GQA"
# Replace the relative path to abs path
ln -s $PROJ_ROOT/playground/data/LLaVA-Pretrain/images playground/data/eval/gqa/data/images
echo "Download eval scripts"
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/eval/gqa/data/eval.zip https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip
cd playground/data/eval/gqa/data
unzip eval.zip
rm eval.zip
cd -
cp llavamini/gqa_eval.py playground/data/eval/gqa/data/eval.py
echo "GQA setup finished"