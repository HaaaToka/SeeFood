
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
remove digits from image_name and change food_label to digit: elipsoid ->1 column->2 irregular->3

"""

dataset_train = pd.read_csv("../train340.csv")
dataset_test = pd.read_csv("../demo.csv")

columns=dataset_test.columns

foods=['apple','egg','lemon','orange','peach','plum','qiwi','tomato','mix',
        'bread','grape','mooncake','sachima','banana','bun','doughnut',
        'fired_dough_twist','litchi','mango','pear']
print(foods.index('apple'))

with open("../demo/rcDemo.csv","w",newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()

    for i in range(dataset_test.shape[0]):
        rw=[]
        for j in range(len(columns)):
            if j==0:
                nm=dataset_test[columns[j]][i].split('0')[0]
                print("TEST",nm)
                rw.append(foods.index(nm))
            elif j==9:
                if dataset_test[columns[j]][i]=='elipsoid':
                    rw.append(1)
                elif dataset_test[columns[j]][i]=='column':
                    rw.append(2)
                else:
                    rw.append(3)
            else:
                rw.append(dataset_test[columns[j]][i])

        writer.writerow({
            'image_name':rw[0], 'Side_coinWidth':rw[1], 'Side_coinHeigth':rw[2], 'Top_coinWidth':rw[3],
           'Top_coinHeigth':rw[4], 'Side_foodWidth':rw[5], 'Side_foodHeigth':rw[6], 'Top_foodWidth':rw[7],
           'Top_foodHeigth':rw[8], 'food_label':rw[9], 'Side_coinForeground_pixel':rw[10],
           'Top_coinForeground_pixel':rw[11], 'Side_foodForeground_pixel':rw[12],
           'Top_foodForeground_pixel':rw[13], 'realVolume':rw[14], 'realDensity':rw[15]
        })
exit()

with open("rcTrain340.csv","w",newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()

    for i in range(dataset_train.shape[0]):
        rw=[]
        for j in range(len(columns)):
            if j==0:
                nm=dataset_train[columns[j]][i].split('0')[0]
                print("TRAIN",nm)
                rw.append(foods.index(nm))
            elif j==9:
                if dataset_train[columns[j]][i]=='elipsoid':
                    rw.append(1)
                elif dataset_train[columns[j]][i]=='column':
                    rw.append(2)
                else:
                    rw.append(3)
            else:
                rw.append(dataset_train[columns[j]][i])

        writer.writerow({
            'image_name':rw[0], 'Side_coinWidth':rw[1], 'Side_coinHeigth':rw[2], 'Top_coinWidth':rw[3],
           'Top_coinHeigth':rw[4], 'Side_foodWidth':rw[5], 'Side_foodHeigth':rw[6], 'Top_foodWidth':rw[7],
           'Top_foodHeigth':rw[8], 'food_label':rw[9], 'Side_coinForeground_pixel':rw[10],
           'Top_coinForeground_pixel':rw[11], 'Side_foodForeground_pixel':rw[12],
           'Top_foodForeground_pixel':rw[13], 'realVolume':rw[14], 'realDensity':rw[15]
        })
