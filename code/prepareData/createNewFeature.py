import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
#totos gokber

dataset=pd.read_csv('../dataset_last.csv')
datasetColumnLength=dataset.shape[0]

calories = { 'apple':0.52, 'banana':0.89, 'bread':3.15, 'bun':2.23,
        'doughnut':4.34, 'egg':1.43, 'fired_dough_twist':24.16,
        'grape':0.69, 'lemon':0.29, 'litchi':0.66, 'mango':0.60,
        'mooncake':18.83, 'orange':0.63, 'peach':0.57, 'pear':0.39,
        'plum':0.46, 'qiwi':0.61, 'sachima':21.45, 'tomato':0.27 }


clVol=dataset['realVolume']
clDens=dataset['realDensity']
names=dataset['image_name']

newColumn=[1]*datasetColumnLength

for i in range(datasetColumnLength):
    newColumn[i]*=clVol[i]*clDens[i]*calories[names[i]]


dataset['real_calorie']=newColumn

dataset.to_csv('../ddgokber.csv',sep=',',encoding='utf-8',index=False)

exit()
"""

dataset=pd.read_csv('../demo/rcDemo.csv')
datasetColumnLength=dataset.shape[0]

foods=['apple','egg','lemon','orange','peach','plum','qiwi','tomato','mix',
        'bread','grape','mooncake','sachima','banana','bun','doughnut',
        'fired_dough_twist','litchi','mango','pear']


calories = { 'apple':0.52, 'banana':0.89, 'bread':3.15, 'bun':2.23,
        'doughnut':4.34, 'egg':1.43, 'fired_dough_twist':24.16,
        'grape':0.69, 'lemon':0.29, 'litchi':0.66, 'mango':0.60,
        'mooncake':18.83, 'orange':0.63, 'peach':0.57, 'pear':0.39,
        'plum':0.46, 'qiwi':0.61, 'sachima':21.45, 'tomato':0.27 }

names=dataset['image_name']

countFood=[0]*len(foods)
average_density=[0]*len(foods)

for elem in dataset.iterrows():
    countFood[int(elem[1][0])]+=1
    average_density[int(elem[1][0])]+=elem[1][-1]

#print(average_density,"\n\n",countFood,"\n")

for i in range(len(foods)):
    if countFood[i]==0:
        continue
    else:
        average_density[i]/=countFood[i]

#print(average_density,"\n\n",countFood)

newColumn=[0]*datasetColumnLength
i=0
for elem in dataset.iterrows():
    newColumn[i] = average_density[int(elem[1][0])]
    i+=1

dataset['average_density']=newColumn

#----------------------------------------------#

newColumn=[0]*datasetColumnLength


i=0
for elem in dataset.iterrows():
    newColumn[i] = calories[foods[int(elem[1][0])]]
    i+=1

dataset['energy']=newColumn


#----------------------------------#

energy=dataset['energy']

newColumn=[0]*datasetColumnLength

clVol=dataset['realVolume']
clDens=dataset['realDensity']

for i in range(datasetColumnLength):
    newColumn[i]=clVol[i]*clDens[i]*calories[foods[names[i]]]

dataset['real_calorie']=newColumn

#----------------------------------#
dataset.to_csv('../demo/rcDemo2.csv',sep=',',encoding='utf-8',index=False)
