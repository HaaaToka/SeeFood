import os
import shutil
import sys
from collections import defaultdict
from math import ceil,floor

dicSide = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[],'mix':[] }
dicTop = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[],'mix':[] }


c=0
for file in os.listdir('../Dataset/Annotations'):
    if 'xml' in file:
        for key in dicSide.keys():
            if key in file:
                if 'S' in file:
                    dicSide[key].append(file)
                else:
                    dicTop[key].append(file)
                break

k='qiwi'
print(dicSide[k][1])
print(dicSide[k][1].replace('S','T'))

t=0
for k in dicTop.keys():
    t+=len(dicSide[k])+len(dicTop[k])
    print(k,"\n",len(dicSide[k]))
    print(len(dicTop[k]),"\n --------------")
print(t)


for key in dicSide:
    "bir daha split gerekirse esit sekilde ayır birinde fazladan eleman varsa fazlaliklari traine salla testte esit sayida olsun"
    count_food_Side = len(dicSide[key])
    train_size_Side=floor((count_food_Side/10)*9.5)+10**10
    print(count_food_Side,train_size_Side,"SIDE>>>>>>>>")
    count_food_Top = len(dicTop[key])
    train_size_Top=floor((count_food_Top/10)*8.5)
    print(count_food_Top,train_size_Top,"TOP>>>>>>>>")

    if(train_size_Side==train_size_Top):

        for i in range(train_size_Side):
            filename=dicSide[key][i]
            shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/train/"+filename)
            filename = filename.replace('S','T')
            shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/train/"+filename)

        for i in range(train_size_Side,count_food_Side):
            filename=dicSide[key][i]
            shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/test/"+filename)
            filename = filename.replace('S','T')
            shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/test/"+filename)

    else:

        lis = dicSide[key]
        dcSi = defaultdict(list)
        for elem in lis:
            name = elem.split('S')[0]
            dcSi[name].append(elem)

        print(dcSi.keys())

        lis2 = dicTop[key]
        dcTo = defaultdict(list)
        for elem in lis2:
            name = elem.split('T')[0]
            dcTo[name].append(elem)


        for kk in dcSi.keys():

            count_food_Side = len(dcSi[kk])
            train_size_Side = floor((count_food_Side/10)*9.5)
            print(count_food_Side,train_size_Side,"SIDE>>>>>>>>",kk)
            count_food_Top = len(dcTo[kk])
            train_size_Top=floor((count_food_Top/10)*9.5)
            print(count_food_Top,train_size_Top,"TOP>>>>>>>>",kk)
            print(kk,dcSi[kk])
            print(kk,dcTo[kk],"\n")

            for i in range(train_size_Side):
                filename=dcSi[kk][i]
                shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/train/"+filename)

            for i in range(train_size_Side,count_food_Side):
                filename=dcSi[kk][i]
                shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/test/"+filename)

            for i in range(train_size_Top):
                filename = dcTo[kk][i]
                shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/train/"+filename)

            for i in range(train_size_Top,count_food_Top):
                filename = dcTo[kk][i]
                shutil.move("/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/"+filename,"/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Annotations/test/"+filename)

































#asd
