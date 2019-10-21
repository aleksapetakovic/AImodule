#Importing Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Multiclass Classification using random forest algorithm, first part same as tensorflow (data prep),
# tends to perform better with fewer data

dataset = pd.read_csv('./outfile.csv')
dataset['allergy'] = dataset['allergy'].map({'yes': 1, 'no': 0})
dataset['medication'] = dataset['medication'].map({'yes': 1, 'no': 0})
dataset['class'] = dataset['class'].map({'low': 0, 'medium': 1, 'high': 2})

categorical_cols = dataset[dataset.columns[5]]
train_dataset = dataset[dataset.columns[0:5]]
bmi_col = dataset[dataset.columns[6]]

categorical_dummies = pd.get_dummies((categorical_cols))
categorical_dummies = categorical_dummies[categorical_dummies.columns[1:4]]

train_dataset = pd.concat([train_dataset,categorical_dummies], axis= 1)
train_dataset = pd.concat([train_dataset, bmi_col], axis=1)

train_labels = dataset[dataset.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(train_dataset, train_labels , train_size = 0.8, random_state =  90)

train_norm = x_train[['age','surgery','docvisit','bmi']]
test_norm = x_test[['age','surgery','docvisit','bmi']]


# Normalize Training Data
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
x_train.update(training_norm_col)
print (x_train.head())
# Normalize Test Data
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns)
x_test.update(testing_norm_col)
print (x_test.head())

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)


def make_prediction_array(age,surgery,docvisit,allergy,medication,cholesterol,diabetes,heart,bmi):
    x_test_norm = norm_record(age,surgery,docvisit,bmi)
    prediction_array = [[x_test_norm[0][0], x_test_norm[0][1],x_test_norm[0][2],allergy,medication,cholesterol,diabetes,heart,x_test_norm[0][3]]]
    print(prediction_array)
    return prediction_array


def norm_record(age,surgery,docvisit,bmi):
    test_norm = np.array([[age,surgery, docvisit, bmi]])
    x_test_norm = std_scale.transform(test_norm)
    return x_test_norm


# Predicting the Test set results
y_pred = classifier.predict(x_test)


def predict(prediction_dataset):
    prediction = classifier.predict(prediction_dataset)
    class_names = ['low', 'medium', 'high']
    return class_names[prediction[0]]


print (accuracy_score(y_test, y_pred))
print(list(zip(train_dataset, classifier.feature_importances_)))
joblib.dump(classifier, 'randomforestmodel.pkl')