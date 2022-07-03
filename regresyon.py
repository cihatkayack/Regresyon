from re import I
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#Retrieving data
data = pd.read_excel('cicekler.xlsx')
 
# Converting type of columns to category
data['renkler']=data['renkler'].astype('category')
data['uzunluk']=data['uzunluk'].astype('category')
data['turler']=data['turler'].astype('category')
 
 
#Assigning numerical values and storing it in another columns
data['renkler_new']=data['renkler'].cat.codes
data['uzunluk_new']=data['uzunluk'].cat.codes
data['turler_new']=data['turler'].cat.codes
 
 
#Create an instance of One-hot-encoder
enc=OneHotEncoder()
 
#Passing encoded columns
enc_data=pd.DataFrame(enc.fit_transform(data[['uzunluk_new','renkler_new','turler_new']]).toarray())
 
#Merge with main
New_df=data.join(enc_data)
New_df.to_excel("New_df.xlsx")
readNew_df = pd.read_excel("New_df.xlsx")

useData = pd.DataFrame({'uzunluk': New_df['uzunluk_new'],'renkler': New_df['renkler_new'], 'turler': New_df['turler_new'], 'boy': data['boy']})

useData.to_excel("data.xlsx")


newData = pd.read_excel("data.xlsx")

FirstData = input("Enter first Data(you need to choose a data from 'cicekler.xlsx'): ")
SecondData = input("Enter second Data(you need to choose a data from 'cicekler.xlsx'): ")

TrainsFirstData = newData[FirstData]
TrainsSecondData = newData[SecondData]

x_train, x_test,y_train,y_test = train_test_split(TrainsFirstData,TrainsSecondData,test_size=0.33, random_state=1234)
lr = LinearRegression()

x_train = pd.DataFrame(x_train)

x_test = pd.DataFrame(x_test)

lr.fit(x_train, y_train)
prediction = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
 
#Grafik çizme
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
 
#Grafik başlığı,x ve y için etiket oluşturma
plt.title("Lineer Regresyon")
plt.xlabel(FirstData)
plt.ylabel(SecondData)

 
