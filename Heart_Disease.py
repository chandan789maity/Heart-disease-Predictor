

import pandas as pd
import numpy as np
df=pd.read_csv("Heart_Disease_Prediction.csv")
x=df.iloc[:,0:13].values
y=df.iloc[:,13].values

#Encoding last column of the dataset
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

#Total Null value showing....
print(df.isnull().sum())

#Splitting the dataset into train and test data.....
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)

#Feature scaling......
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Applying K-NN machine algorithm to train the model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(weights='distance')
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

#Getting the confusion matrix ready.
from sklearn.metrics import confusion_matrix
print("This is my confusion matrix :"
      ,confusion_matrix(y_test, y_pred))

#Getting the accuracy score ready.
from sklearn.metrics import accuracy_score
print("This is the my model using KNN:")
print(accuracy_score(y_test, y_pred))

#Decision Tree......
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)

#Confusion matrix......
from sklearn.metrics import confusion_matrix
print("Decision tree confusion matrix: ",confusion_matrix(y_test, y_pred))

#Accuracy Score
from sklearn.metrics import accuracy_score
print("Decision Tree accuracy score:")
print(accuracy_score(y_test, y_pred))

#Random Forest.....
from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0)
rc.fit(x_train,y_train)
y_pred=rc.predict(x_test)

#Confusion matrix....
from sklearn.metrics import confusion_matrix
print("Random forest confusion matrix: ",confusion_matrix(y_test, y_pred))

#Accuracy Score
from sklearn.metrics import accuracy_score
print("Random forest accuracy score:")
print(accuracy_score(y_test, y_pred))

#Applying the SVM Classifier
from sklearn.svm import SVC
sv=SVC(kernel='sigmoid',random_state=0)
sv.fit(x_train,y_train)
y_pred3=sv.predict(x_test)

#Getting the confusion matrix ready.
from sklearn.metrics import confusion_matrix
print("SVM confusion matrix :"
      ,confusion_matrix(y_test, y_pred3))

#Getting the accuracy score ready.
from sklearn.metrics import accuracy_score
print("SVM Accuracy Score:")
print(accuracy_score(y_test, y_pred3))

#Getting the predictor ready.
input=(57,1,2,124,261,0,0,141,0,0.3,1,0,7)
input_array=np.asarray(input);
input_reshaped=input_array.reshape(1,-1);
std_data=sc.transform(input_reshaped);
#print(std_data);

#Applying the random forest classifier as this has the greatest accuracy score in this case.
prediction=rc.predict(std_data);
if(prediction):
    print("Heart disease is present!!!");
else:
    print("Heart disease is not present!!!");