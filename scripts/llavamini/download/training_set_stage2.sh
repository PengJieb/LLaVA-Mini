###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2025-02-18 16:25:03
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-02-24 17:13:40
 # @FilePath: /LLaVA-Mini/scripts/llavamini/download_visual_instruct_tuning_datasets.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# setup proxy, not necessary
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890

# Before running this scripts, install aria2
# The install command: conda install -c conda-forge aria2 -y

# aria2 parameters
MAXIMUM_CONNECTION_PER_SERVER=16
DOWNLOAD_CONNECTIONS=16

# specify weather download the dataset
DOWNLOAD_COCO=0
DOWNLOAD_GQA=0
DOWNLOAD_OCR_VQA=1
DOWNLOAD_VG=0
DOWNLOAD_TEXTVQA=0

if [ $DOWNLOAD_COCO -eq 1 ]; then
echo "Downloading the COCO dataset..."
mkdir playground/data/coco
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/coco/train2017.zip http://images.cocodataset.org/zips/train2017.zip 
echo "COCO dataset download finished"
cd playground/data/coco
unzip train2017.zip
rm train2017.zip
cd -
fi

if [ $DOWNLOAD_GQA -eq 1 ]; then
echo "Downloading the GQA dataset..."
mkdir playground/data/gqa
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/gqa/train2017.zip https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
echo "GQA dataset download finished"
cd playground/data/gqa
unzip train2017.zip
rm train2017.zip
cd -
fi

if [ $DOWNLOAD_OCR_VQA -eq 1 ]; then
echo "Downloading OCR-VQA dataset..."
mkdir playground/data/ocr_vqa
gdown --no-check-certificate https://drive.google.com/uc?id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q -O playground/data/ocr_vqa/dataset.json
# copy modified download scripts: enhanced by error check and multi-process download
cp llavamini_scaling/loadDataset.py playground/data/ocr_vqa
cd playground/data/ocr_vqa
python loadDataset.py
cd -
echo "OCR-VQA dataset download finished"
fi

if [ $DOWNLOAD_VG -eq 1 ]; then
echo "Downloading VisualGenome dataset..."
mkdir playground/data/vg
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/vg/vg_100k.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/vg/vg_100k_2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
echo "VisualGenome dataset download finished"
cd playground/data/vg
unzip vg_100k_2.zip
unzip vg_100k.zip
rm vg_100k.zip
rm vg_100k_2.zip
cd -
fi

if [ $DOWNLOAD_TEXTVQA -eq 1 ]; then
echo "Downloading TextVQA dataset..."
mkdir playground/data/textvqa
aria2c -x $MAXIMUM_CONNECTION_PER_SERVER -s $DOWNLOAD_CONNECTIONS -c -o playground/data/textvqa/train_val_images.zip https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
echo "TextVQA dataset download finished"
cd playground/data/textvqa
unzip train_val_images.zip
rm train_val_images.zip
cd -
fi






