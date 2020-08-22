import pandas as pd 
import requests
import statsmodels.api
import numpy as np
from sklearn.linear_model import LogisticRegression


read = pd.read_csv('/Users/karimwolbert/titanic/train.csv') # requests the data from csv file

df = pd.DataFrame(read) # putting data into a dataframe

df['Sex'] = df['Sex'].replace(['male'],0) # changes the string male into 0
df['Sex'] = df['Sex'].replace(['female'],1) # changes the string female into 1

sex = df['Sex'] # putting the columm Sex in a variable for direct access

all_passengers = len(sex)
count_female = sex.sum() # summed up number of femnale passengers
count_male = len(sex) - count_female # summed up number of male passengers

# Prediction of chance to survive by sex

survived = df['Survived']
age = df['Age']
fare = df['Fare']

sex_and_fare = np.array([sex, fare])
result_survive = statsmodels.discrete.discrete_model.Probit(survived, sex)
result_survive_2 = result_survive.fit()
probit_margeff = result_survive_2.get_margeff()
print(result_survive_2.summary())
print(probit_margeff.summary())
print(result_survive_2.params) # gives the coefficient







