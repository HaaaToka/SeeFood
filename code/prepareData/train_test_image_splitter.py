import os
import shutil
import sys
from math import ceil,floor


dic = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[],'mix':[] }


for file in os.listdir('/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/test'):
    te = file.split('.')[0]+'.JPG'
    shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/JPEGImages/"+te,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/JPEGImages/test/"+te)


for file in os.listdir('/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/train'):
    te = file.split('.')[0]+'.JPG'
    shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/JPEGImages/"+te,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/JPEGImages/train/"+te)
