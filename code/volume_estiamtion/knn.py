import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('mix2csv.csv')

dataset = dataset.drop(dataset[dataset['Top_coinForeground_pixel'] < 1000].index)
dataset = dataset.drop(dataset[dataset['Side_coinForeground_pixel'] < 1000].index)
dataset = dataset.drop(dataset[dataset['Top_foodForeground_pixel'] < 1000].index)
dataset = dataset.drop(dataset[dataset['Side_foodForeground_pixel'] < 1000].index)

dataset["top_ratio"] = (dataset['Top_foodHeigth'] * dataset['Top_foodWidth']) / (dataset['Top_coinHeigth'] * dataset['Top_coinWidth'])
dataset['side_ratio'] = (dataset['Side_foodWidth'] * dataset['Side_foodHeigth']) / (dataset['Side_coinHeigth'] * dataset['Side_coinWidth'])
dataset['top_ratio_pixels'] = (dataset['Top_foodForeground_pixel'] / dataset['Top_coinForeground_pixel'])
dataset['side_ratio_pixels'] = (dataset['Side_foodForeground_pixel'] / dataset['Side_coinForeground_pixel'])

dataset['top_coin_ratio'] = 2.5 / ((dataset['Top_coinWidth'] + dataset['Top_coinHeigth']) )
dataset['side_coin_ratio'] = 2.5 / ((dataset['Side_coinWidth'] + dataset['Side_coinHeigth']))

dataset['top_area'] = (dataset['Top_foodHeigth'] * dataset['Top_foodWidth']) * dataset['top_coin_ratio'] * dataset['top_ratio_pixels']
dataset['side_area'] = (dataset['Side_foodHeigth'] * dataset['Side_foodWidth']) * dataset['side_coin_ratio'] * dataset['side_ratio_pixels']

#dataset['kcal'] = dataset['real_calorie'] / (dataset['realDensity'] * dataset['realVolume'])


"""dataset_train = pd.read_csv("rcTrain340.csv")
dataset_test = pd.read_csv("rcTest340.csv")

#dataset_test = dataset_test.drop('image_name',axis=1)
#dataset_train = dataset_train.drop('image_name',axis=1)

print(dataset_train.head())
print(dataset_test.head())

y_train=dataset_train.iloc[:,0].values
X_train=dataset_train.iloc[:,1:].values
y_test=dataset_test.iloc[:,0].values
X_test=dataset_test.iloc[:,1:].values
"""

from sklearn.model_selection import train_test_split
dataset_train,dataset_test = train_test_split(dataset,test_size=0.3,random_state=42)

test_realCalori=dataset_test['real_calorie'].tolist()
test_reDens = dataset_test['realDensity'].tolist()
test_energy=dataset_test['energy'].tolist()
test_avDens = dataset_test['average_density'].tolist()

dicTrain = { 'apple':0, 'banana':0, 'bread':0, 'bun':0,
        'doughnut':0, 'egg':0, 'fired_dough_twist':0,
        'grape':0, 'lemon':0, 'litchi':0, 'mango':0,
        'mooncake':0, 'orange':0, 'peach':0, 'pear':0,
        'plum':0, 'qiwi':0, 'sachima':0, 'tomato':0 }
dicTest = { 'apple':0, 'banana':0, 'bread':0, 'bun':0,
        'doughnut':0, 'egg':0, 'fired_dough_twist':0,
        'grape':0, 'lemon':0, 'litchi':0, 'mango':0,
        'mooncake':0, 'orange':0, 'peach':0, 'pear':0,
        'plum':0, 'qiwi':0, 'sachima':0, 'tomato':0 }

foods=['apple','egg','lemon','orange','peach','plum','qiwi','tomato','mix',
        'bread','grape','mooncake','sachima','banana','bun','doughnut',
        'fired_dough_twist','litchi','mango','pear']

print(dataset_train,"X_train")

for x_t in dataset_train.iterrows():
    #print(x_t,len(x_t[1]), x_t[1][0])
    dicTrain[foods[int(x_t[1][0])]]+=1
print(dicTrain)

for x_t in dataset_test.iterrows():
    #print(x_t,len(x_t[1]), x_t[1][0])
    dicTest[foods[int(x_t[1][0])]]+=1
print(dicTest)

names=dicTest.keys()

groupCount=len(names)
fig,ax=plt.subplots()
index=np.arange(groupCount)
bar_width = 0.35
opacity = 0.8

countTest=[]
countTrain=[]
for elem in names:
    countTest.append(dicTest[elem])
    countTrain.append(dicTrain[elem])


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
#plt.show()

Gokberk_useless_features = ['realVolume', 'realDensity', 'Top_coinWidth',
                            'Top_coinHeigth', 'Side_coinWidth', 'Side_coinHeigth',
                            'real_calorie']
Ben_useless_features = ['realVolume','realDensity','real_calorie']
X_train = dataset_train.drop(Gokberk_useless_features,axis=1)
y_train = dataset_train['realVolume']
X_test = dataset_test.drop(Gokberk_useless_features,axis=1)
y_test = dataset_test['realVolume']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

print(X_train)

from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error

rmse_val = [] #to store rmse values for different k
k_value=50
preL=[]
for K in range(1,k_value):

    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    preL.append(pred)
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

plt.figure(figsize=(12,6))
plt.plot(range(1,k_value),rmse_val,color='red',linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('RMSE Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
#plt.show()

mnK = rmse_val.index(min(rmse_val))
print("MNK KKKKKK",mnK)

model = neighbors.KNeighborsRegressor(n_neighbors=mnK+1)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("ERRORRRRRR ->>", sqrt(mean_squared_error(y_test,pred)) )

countFood=[0]*len(foods)
meanRealVol=[0]*len(foods)
meanEstiamteVol=[0]*len(foods)

meanRealCal=[0]*len(foods)
meanEstiamteCal=[0]*len(foods)

imna = dataset_test['image_name'].tolist()
reVol = y_test.tolist()

pred_calorie=[0]*len(pred)

for i in range(len(imna)):

    pred_calorie[i]=test_avDens[i]*test_energy[i]*pred[i]
    meanRealCal[imna[i]]+=test_realCalori[i]
    meanEstiamteCal[imna[i]]+=pred_calorie[i]

    countFood[imna[i]] += 1
    meanRealVol[imna[i]]+=reVol[i]
    meanEstiamteVol[imna[i]]+=pred[i]

#for i in range(len(pred)):
#    print(i,"->",foods[imna[i]],pred[i],"-",reVol[i],"-",test_reDens[i])
#print("####################################################orange")
#for i in range(len(pred_calorie)):
#    print(i,"->",foods[imna[i]],pred_calorie[i],test_realCalori[i])


print("##########",sqrt(mean_squared_error(test_realCalori,pred_calorie)))

yV,yC=0,0
print("food type & Estimation Food Count & Mean Volume & Mean Estimation Volume & Volume Error(\%)\\\\\n \\hline")
for i in range(len(foods)):
    if countFood[i]!=0:
        yV+=abs(((100*meanEstiamteVol[i])/meanRealVol[i])-100)
        yC+=abs(((100*meanEstiamteCal[i])/meanRealCal[i])-100)


        meanRealVol[i]/=countFood[i]
        meanEstiamteVol[i]/=countFood[i]
        print(foods[i],"&",countFood[i],"&",float("{0:.2f}".format(meanRealVol[i])),
              "&",float("{0:.2f}".format(meanEstiamteVol[i])),"&",
              float("{0:.2f}".format(((100*meanEstiamteVol[i])/meanRealVol[i])-100)), "\\\\\n\\hline")

print("***************************************")

print("food type & Estimation Food Count & Mean Calaori & Mean Estimation Calorie & Mean Estimation Calorie Error(\%) \\\\\n \\hline")
for i in range(len(foods)):
    if countFood[i]!=0:

        meanRealCal[i]/=countFood[i]
        meanEstiamteCal[i]/=countFood[i]

        print(foods[i],"&",countFood[i],
              "&",float("{0:.2f}".format(meanRealCal[i])),"&",float("{0:.2f}".format(meanEstiamteCal[i])),
               "&",float("{0:.2f}".format(((100*meanEstiamteCal[i])/meanRealCal[i])-100)), "\\\\\n\\hline")



from pandas.plotting  import parallel_coordinates
plt.figure(figsize=(15,10))
parallel_coordinates(dataset_train,'image_name')
plt.title("Paralel Cordinates Plot")
plt.xlabel('Features')
plt.ylabel('Feature Values')
plt.legend(loc=1,prop={'size':15},frameon=True,shadow=True,facecolor='white',edgecolor='black')
plt.show()


from pandas.plotting import andrews_curves
plt.figure(figsize=(15,10))
andrews_curves(dataset_train,'image_name')
plt.title("Paralel Cordinates Plot")
plt.legend(loc=1,prop={'size':15},frameon=True,shadow=True,facecolor='white',edgecolor='black')
plt.show()



"""print(dataset_test.shape,dataset_train.shape)
print(yV,yV/19,yC,yC/19)
print("---->>>>>",sqrt(sum(meanEstiamteVol)/19),sqrt(sum(meanEstiamteCal)/19))
"""

"""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
k_value=40
preL=[]
for i in range(1, k_value):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    preL.append(pred_i)
    error.append(np.mean(pred_i != y_test))

print(len(preL),len(preL[0]),k_value)
for i in range(len(preL[0])):
    print(y_test[i]," -->>" ,end=' ')
    for k in  range(k_value-1):
        print(preL[k][i],end=' ')
    print("\n")

plt.figure(figsize=(12, 6))
plt.plot(range(1, k_value), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()









I am gonna talk about our food dataset.
There are almost three thousand images in the dataset.
Every image contains one or two food and one China coin but we use images that contain only one food.
Every food taken from two different positions, one from the side and one from the top.

The dataset provides real volume, mass, and labeled images' xml file.
We are using labeled images' xml file when we train our object detection model.
We need real volume and mass to calculate our estimating model's error.

We used two thousand eight hundred  seventy images in total but we split our data two times.
Firstly, we splitted it for object detection.
We used two thousand five hundred  fifty eight images for training.
Rest of the images were used for testing. We saw our test result were good.
Then we obtained all images numerical form with using object detection model and GrabCut.
Secondly, we splitted all images' numerical form to estimate volume.
We used two thousand two images for training.
Rest of the images were used for testing.
Then we are calculating calorie with simple mathematical formulas.
------------------

In our dataset, we have nineteen kinds of food. You see all of them. sayarsın burda yiyecekelri
------------------

There are two main ways to predict calories.
Firstly we estimated the volume then we estimated the calories using the volume

However, we discovered that estimating the calories directly was giving us many accurate results.
-------------------

We used 2 different models for volume estimation. The first is K Nearest Neighbors and second Random Forest.
-----------------

KNN can be used for both classification and regression.
Our problem is estimating volume. So it is a regression problem.
Let's look at how does KNN work for regression? It is very easy but very expensive for the time.
Firstly we choose nearest K neighbors.
The estimated value is the average of the values of the nearest K neighbors.
---------------

How do we find the best value of K?
We used the Root Mean Square Error method.
We calculate RMSE for the values of K that is from one to fifty.
The minimum RMSE value is best K value. K is 6 for our the dataset.
---------------

KNN's volume estimation result was surprisingly satisfying us.
For example, a mean real volume of apple is three hundred twenty-one and knn's mean estimation volume is three hundred twenty-five.
It is relatively good. Same results for the calorie.

Another method is Random forest. My friend Gokberk is going to talking about it.
----------------





"""
