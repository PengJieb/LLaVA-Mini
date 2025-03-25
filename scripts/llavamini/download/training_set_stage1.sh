###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2025-02-18 16:14:19
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-03-24 16:33:07
 # @FilePath: /LLaVA-Mini/scripts/llavamini/download_llava_pretrain_dataset.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# setp mirror of huggingface, not necessary
# export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading the LLaVA-Pretrain dataset..."
huggingface-cli download --repo-type dataset --resume-download liuhaotian/LLaVA-Pretrain --local-dir playground/data --local-dir-use-symlinks False
echo "LLaVA-Pretrain dataset download finished"

huggingface-cli download --repo-type dataset --resume-download liuhaotian/LLaVA-Instruct-150K --local-dir playground/data/LLaVA-Instruct-150K --local-dir-use-symlinks False

echo "unzip the dataset..."
mkdir playground/data/LLaVA-Pretrain
mkdir playground/data/LLaVA-Pretrain/images
cd playground/data/LLaVA-Pretrain/images
mv ../../images.zip .
unzip images.zip
rm images.zip
cd -
echo "unzip finished"

