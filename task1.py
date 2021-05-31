import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy
import os

os.system("tput setaf 3")
print("\t  \t \t welcome to ML Salary Predictor App")
print()

db = pd.read_csv('salary.csv')
y = db['Salary']
x = db['YearsExperience']
x = x.values.reshape(30,1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

os.system("tput setaf 2")
z = float(input("enter years of experience: "))
out = model.predict([[ z ]])

os.system("tput setaf 6")
print("The estimated Salary is >>>",out)

os.system("tput setaf 7")