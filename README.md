# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Abisheik raj.J
RegisterNumber: 212224230006
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

### DATASET INFO
![image](https://github.com/user-attachments/assets/a20b9cbc-7d29-4e1f-bb4f-74ad7ca4bf50)

### NULL DATASET
![image](https://github.com/user-attachments/assets/af0c39bc-eed1-408c-b2cf-2a012bb0f617)

### VALUE COUNT IN LEFT COLUMN
![image](https://github.com/user-attachments/assets/81204c17-13bb-49d9-a273-0eede8810e3f)

### DATASET TRANSFORMED HEAD
![image](https://github.com/user-attachments/assets/ed34822f-7e4d-4b63-9429-6ad404c0e76c)

### X.HEAD
![image](https://github.com/user-attachments/assets/b0a53305-07db-44a6-aab7-d971957ab108)

### ACCURACY
![image](https://github.com/user-attachments/assets/d03511b9-1f67-45ea-b0d0-952777d09acb)

### DATA PREDICTION
![image](https://github.com/user-attachments/assets/aabb3383-7b9a-49ba-95e0-cea4aab4380c)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
