import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dicTrain = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired_dough_twist':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[] }
dicTest = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired_dough_twist':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[] }



foods=['apple','egg','lemon','orange','peach','plum','qiwi','tomato','mix',
        'bread','grape','mooncake','sachima','banana','bun','doughnut',
        'fired_dough_twist','litchi','mango','pear']


mixTest,mixTrain=0,0
for elem in os.listdir("../Dataset/JPEGImages/test"):
    if 'mix' in elem:
        mixTest+=1
        continue
    for keys in dicTest.keys():
        if keys in elem:
            dicTest[keys].append(elem)
            break


for elem in os.listdir("../Dataset/JPEGImages/train"):
    if 'mix' in elem:
        mixTrain+=1
        continue
    for keys in dicTest.keys():
        if keys in elem:
            dicTrain[keys].append(elem)
            break

print(mixTest,mixTrain)

names=dicTest.keys()

groupCount=len(names)
fig,ax=plt.subplots()
index=np.arange(groupCount)
bar_width = 0.35
opacity = 0.8

countTest=[]
countTrain=[]
for elem in names:
    countTest.append(len(dicTest[elem]))
    countTrain.append(len(dicTrain[elem]))


rects1 = plt.barh(index, countTest, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Test')

rects2 = plt.barh(index + bar_width, countTrain, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Train')

ax.set_yticks(index)
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel('Counts')
ax.set_title("Counts of Used Food")
plt.legend()

plt.tight_layout()
plt.show()
