import csv
import os
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

"""
filename = 'InsuranceData.txt'
with open(filename) as infile, open('outfile.csv','w') as outfile:
    for line in infile:
        outfile.write(line.replace(' ',','))
"""


dataset = pd.read_csv('./outfile.csv')
dataset['allergy'] = dataset['allergy'].map({'yes': 1, 'no': 0})
dataset['medication'] = dataset['medication'].map({'yes': 1, 'no': 0})
dataset['class'] = dataset['class'].map({'low': 0, 'medium': 1, 'high': 2})

categorical_cols = dataset[dataset.columns[5]]
train_dataset = dataset[dataset.columns[0:5]]
bmi_col = dataset[dataset.columns[6]]
categorical_dummies = pd.get_dummies((categorical_cols))
categorical_dummies = categorical_dummies[categorical_dummies.columns[0:3]]
train_dataset = pd.concat([train_dataset,categorical_dummies], axis= 1)
train_dataset = pd.concat([train_dataset, bmi_col], axis=1)
print(train_dataset)
train_labels = dataset[dataset.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(train_dataset, train_labels , train_size = 0.9, random_state = 2)
y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)
#print(y_binary_train)
train_norm = x_train[['age','surgery','docvisit','bmi']]
test_norm = x_test[['age','surgery','docvisit','bmi']]

# Normalize Training Data
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
x_train.update(training_norm_col)
print (x_train.head())
# normalize test data
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns)
x_test.update(testing_norm_col)
print (x_test.head())


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True, input_shape=[len(train_dataset.keys())]),  # input shape required


    tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True),
    tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True),

    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# mean absolute error and mean squared error
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#   sparse_categorical_crossentropy or hinge for training with risk as 0, 1, 2,
#   if one-hotted use categorical_crossentropy (output shape is 3)
checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(x_train,y_binary_train, validation_data=(x_test, y_binary_test) ,callbacks = [cp_callback],epochs=200)
