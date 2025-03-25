import json
import sys
import os
import urllib.request as ureq
import pdb
import pathlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

download=1 # 0 if images are already downloaded

###############################################################
######################### load dataset json file ###############
################################################################
with open('dataset.json', 'r') as fp:
        data = json.load(fp)

## dictionary data contains image URL, questions and answers ##




################################################################
############### Script for downloading images ##################
################################################################
## Make a directory images to store all images there ##########
# if download == 1:
#     pathlib.Path('./images').mkdir(exist_ok=True)
#     for k in tqdm(data.keys()):
#         ext=os.path.splitext(data[k]['imageURL'])[1]
#         outputFile='images/%s%s'%(k,ext)
#         # pdb.set_trace()
#         ureq.urlretrieve(data[k]['imageURL'],outputFile)    


download_dir = './images'
report_intermedia_second=10
download_threads=8

if download == 1:
    pathlib.Path(download_dir).mkdir(exist_ok=True)

def download_image(k, image_url):
    ext = os.path.splitext(image_url)[1]
    output_file = f'{download_dir}/{k}{ext}'
    if pathlib.Path(output_file).exists():
        return k  # Return the key if the file already exists
    ureq.urlretrieve(image_url, output_file)
    return k  # Return the key for tracking

def scan_downloaded_images():
    while True:
        time.sleep(report_intermedia_second)  # Wait for 10 seconds
        num_images = len([name for name in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, name))])
        print(f"Downloaded images count: {num_images}")

# Start a thread to scan the download directory
scanner_thread = threading.Thread(target=scan_downloaded_images, daemon=True)
scanner_thread.start()

with ThreadPoolExecutor(max_workers=download_threads) as executor:  # You can adjust max_workers
    futures = {executor.submit(download_image, k, data[k]['imageURL']): k for k in data.keys()}
    for future in tqdm(as_completed(futures), total=len(futures)):
        k = futures[future]
        try:
            future.result()  # Raise exception if download failed
            # print(f"Downloaded {k}")
        except Exception as e:
            print(f"Failed to download {k}: {e}")


#################################################################
################### Example of data access #####################
################################################################
for k in data.keys():
    ext=os.path.splitext(data[k]['imageURL'])[1]
    imageFile='images/%s%s'%(k,ext)

    print('************************')
    print('Image file: %s'%(imageFile))
    print('List of questions:')
    print(data[k]['questions'])
    print('List of corresponding answers:')
    print(data[k]['answers'])
    print('Use this image as training (1), validation (2) or testing (3): %s'%(data[k]['split']))
    print('*************************')





######################################################################
########################### Get dataset stats ########################
######################################################################
genSet=set()
for k in data.keys():
    genSet.add(data[k]['genre'])



numImages=len(data.keys())
numQApairs=0
numWordsInQuestions=0
numWordsInAnswers=0
numQuestionsPerImage=0
ANS=set() # Set of unique answers
authorSet=set()
bookSet=set()


for imgId in data.keys():
    numQApairs = numQApairs+len(data[imgId]['questions'])
    numQuestionsPerImage = numQuestionsPerImage + len(data[imgId]['questions'])
    authorSet.add(data[imgId]['authorName'])
    bookSet.add(data[imgId]['title'])

    for qno in range(len(data[imgId]['questions'])):
        ques=data[imgId]['questions'][qno]
        numWordsInQuestions = numWordsInQuestions+len(ques.split())
    for ano in range(len(data[imgId]['answers'])):
        ans=data[imgId]['answers'][ano]
        ANS.add(ans)
        numWordsInAnswers = numWordsInAnswers+len(str(ans).split())



print("--------------------------------")
print("Number of Images: %d" %(numImages))
print("Number of QA pairs: %d" %(numQApairs))
print("Number of unique author: %d" %(len(authorSet)))
print("Number of unique title: %d" %(len(bookSet)))
print("Number of unique answers: %d" %(len(ANS)))
print("Number of unique genre: %d" %(len(genSet)))
print("Average question length (in words): %.2f" %(float(numWordsInQuestions)/float(numQApairs)))
print("Average answer length (in words): %.2f" %(float(numWordsInAnswers)/float(numQApairs)))
print("Average number of questions per image: %.2f" %(float(numQuestionsPerImage)/float(numImages)))
print("--------------------------------")

