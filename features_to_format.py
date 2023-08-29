import numpy as np  
import pandas as pd 

# for training

data = pd.read_csv('./BAA/training_data/train_features.csv')
lst= list(range(1002))
inx= list(range(len(data)))
x= pd.DataFrame(index= inx, columns= lst)
k = data.iat[0, 1][slice(2, len(data.iat[0, 1])-2)].split()[0]
for i in range(len(data)):
    x.iat[i,0]= int(data.iat[i,0][:len(data.iat[i,0])-4])
    for j in range(1, 1001):
        x.iat[i,j]= np.float64(data.iat[i,1].split("[")[3].split("]")[0].split(',')[j-1].strip())
    x.iat[i, 1001]= np.float(data.iat[i,2])
    if(i%1000==0):
        print(i)
x.to_csv('final_train_features.csv', index=False, header=False)

# for val

data = pd.read_csv('./BAA/validation_data/val_features.csv')
lst = list(range(1002))
inx = list(range(len(data)))
x = pd.DataFrame(index=inx, columns=lst)
k = data.iat[0, 1][slice(2, len(data.iat[0, 1])-2)].split()[0]
for i in range(len(data)):
    x.iat[i,0]= int(data.iat[i,0][:len(data.iat[i,0])-4])
    for j in range(1, 1001):
        x.iat[i,j]= np.float64(data.iat[i,1].split("[")[3].split("]")[0].split(',')[j-1].strip())
    x.iat[i, 1001]= np.float(data.iat[i,2])
    if(i%1000==0):
        print(i)
x.to_csv('final_val_features.csv', index= False, header= False)
# for test

data = pd.read_csv('./BAA/testing_data/test_features.csv')
lst = list(range(1002))
inx = list(range(len(data)))
x = pd.DataFrame(index=inx, columns=lst)
k = data.iat[0, 1][slice(2, len(data.iat[0, 1])-2)].split()[0]
for i in range(len(data)):
    x.iat[i,0]= int(data.iat[i,0][:len(data.iat[i,0])-4])
    for j in range(1, 1001):
        x.iat[i,j]= np.float64(data.iat[i,1].split("[")[3].split("]")[0].split(',')[j-1].strip())
    x.iat[i, 1001]= np.float(data.iat[i,2])
    if(i%1000==0):
        print(i)
x.to_csv('final_test_features.csv', index=False, header=False)
