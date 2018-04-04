


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
Y = dataset.iloc[:,1].values
#independent variables for prediction
#indep_variables= [ 'Pclass', 'Sex', 'Age','Parch', 'Fare', 'SibSp']
#Xone= dataset[indep_variables]
#X= Xone.iloc[:,:].values

#visualizing variable graphically
#Survived
dataset.Survived.value_counts().plot(kind='bar', alpha=0.5)
#Survived according to age
plt.scatter(dataset.Survived, dataset.Age, alpha=0.1)
plt.title("Age vs survived")
#Pclass
dataset.Pclass.value_counts().plot(kind='bar', alpha=0.5)
plt.title("class distribution")
#male population
dataset.Survived[dataset.Sex=="male"].value_counts(normalize=True).plot(kind="bar", alpha= 0.5)
plt.title("Male passengers")
#female passengers
dataset.Survived[dataset.Sex=="female"].value_counts(normalize=True).plot(kind="bar", alpha= 0.5)
plt.title("Female passengers")

def clean_data(df):
    #independent variables for prediction
    indep_variables= [ 'Pclass', 'Sex', 'Age','Parch', 'Fare', 'SibSp','Embarked']
    Xone= df[indep_variables]
    K= Xone.iloc[:,:].values
# Sex is a categorical variable
    # Encoding Sex variable and Embarked variable
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    labelencoder_X = LabelEncoder()
    K[:, 1]=labelencoder_X.fit_transform(K[:, 1])
    labelencoder_x=LabelEncoder()
    K[:,6]=labelencoder_x.fit_transform(K[:,6])
    onehot=OneHotEncoder(categorical_features=[6])
    K=onehot.fit_transform(K).toarray()
    
    #K[:, 6]=labelencoder_XX.fit_transform(K[:, 6])
    #Age variable has Nan values in the data set, these are replaced with mean
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
    imputer=imputer.fit(K[:, :])
    K[:,:]= imputer.transform(K[:, :])
    
    #Parch and Sibsp both give info about family members 
    #These two variables are similar SO can be added
    #K[:, 4]=K[:, 4]+K[:, 6]
    
    #K=K[:, [0,1,2,3,4,5,7]]
    
    return K
    
X=clean_data(dataset)
#fitting random forest regression to data set
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=200)
regressor.fit(X, Y)


df_test = pd.read_csv('test.csv')
X_test = clean_data(df_test)

Y_pred = regressor.predict(X_test.astype(float))
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('motanikivachai.csv', index=False)


    


