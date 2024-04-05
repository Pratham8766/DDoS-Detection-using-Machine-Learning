import pandas as pd
import numpy as np
import missingno as msno

df = pd.read_csv("sdn_dataset.csv")

df.head(10)

print("This dataset has " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns.")

df.info()

#Data Preprocessing
#Analysis of Missing Data Distribution in the Dataset
msno.matrix(df)

#Frequency of Null Values in the Columns
df.isnull().sum()

#Drop rows with null values
df.dropna(inplace = True)

df.isnull().sum()

#Dataframe after removing Null Values
print("This dataset has now " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns after dropping Null Values.")

#Features in the Dataset
# 1. Categorical Features
categorical_features = [
    feature for feature in df.columns
        if(df[feature].dtypes == 'O')
]

print("Categorical features summary: \n")
print("Total number of Categorical features: ", len(categorical_features))
print("List of Categorical features:")
for i, feature in enumerate(categorical_features, 1):
    print(f"{i}. {feature}")

# 2. Numerical Features
numerical_features = [
    feature for feature in df.columns
        if(df[feature].dtypes != 'O')
]

print("Numerical features summary: \n")
print("Total number of Numerical features: ", len(numerical_features))
print("List of Numerical features:")
for i, feature in enumerate(numerical_features, 1):
    print(f"{i}. {feature}")

#Distinct Values in Numerical Features
df[numerical_features].nunique(axis = 0)

# 2.1 Discrete Numerical Features
discrete_numerical_features = [
    feature for feature in numerical_features
        if(df[feature].nunique() <= 15 and feature != 'label')
]

print("Discrete numerical features summary: \n")
print("Total number of Discrete numerical features: ", len(discrete_numerical_features))
print("List of Discrete numerical features:")
for i, feature in enumerate(discrete_numerical_features, 1):
    print(f"{i}. {feature}")

# 2.2 Continuous Features
continuous_features = [
    feature for feature in numerical_features
        if( feature not in discrete_numerical_features + ['label'])
]

print("Continuous features summary: \n")
print("Total number of Continuous features: ", len(continuous_features))
print("List of Continuous features:")
for i, feature in enumerate(continuous_features, 1):
    print(f"{i}. {feature}")

# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting Functions
def countPlotDistribution(col):
    sns.set_theme(style = "darkgrid")
    sns.countplot(y = col, data = df, palette = "Set2", hue = col, legend = "auto").set(title = "Distribution of "+ col)

def histoPlotDistribution(col):
    sns.set_theme(style = "darkgrid")
    sns.histplot(data = df, x = col, kde = True, color = "red").set(title = "Distribution of "+ col)

#Distribution of Categorical features
f = plt.figure(figsize = (8, 20))
for i in range(len(categorical_features)):
    f.add_subplot(len(categorical_features), 1, i+1)
    countPlotDistribution(categorical_features[i])
plt.show()

# Distribution of Discrete Numerical Features
for feature in discrete_numerical_features:
    plt.figure(figsize = (8, 4))
    cat_num = df[feature].value_counts()
    sns.barplot(x = cat_num.index, y = cat_num, palette = "Set2", legend = False, hue = cat_num.index).set(title = feature, ylabel = "Frequency", xlabel = "")
    plt.show()

# Distribution of Continuous Features
f = plt.figure(figsize = (20, 90))
for i in range(len(continuous_features)):
    f.add_subplot(len(continuous_features), 2, i+1)
    histoPlotDistribution(continuous_features[i])
plt.show()

# Frequency distribution of Label feature
benign = df[df['label'] == 0]
malign = df[df['label'] == 1]

print("Percentage of DDOS attack that has not occured: {:.2f}%".format((len(benign) / df.shape[0]) * 100))
print("Percentage of DDOS attack that has occured: {:.2f}%".format((len(malign) / df.shape[0]) * 100))

# Distribution of Label Feature [Malign vs Benign]
labels = ['benign', 'malign']
classes = pd.value_counts(df['label'], sort = True) / df['label'].count() * 100
classes.plot(kind = "bar", color = ['blue', 'red'])
plt.title("Label class Distribution")
plt.xticks(range(2), labels)
plt.xlabel("Label")
plt.ylabel("Frequency in %")

def get_malign_protocols_percentage():
    arr = [x for x, y in zip(df['Protocol'], df['label']) if (y == 1)]
    perc_arr = []
    for i in ['UDP', 'TCP', 'ICMP']:
        perc_arr.append(arr.count(i) / len(arr) * 100)
    return perc_arr

# Protocol distribution for Malign attacks
pie_fig, ax1 = plt.subplots(figsize = [7, 7])
ax1.pie(get_malign_protocols_percentage(), explode = (0.1, 0, 0), autopct = "%1.1f%%", shadow = True, startangle = 90)
ax1.axis("equal")
ax1.legend(['UDP', 'TCP', 'ICMP'], loc = "best")
plt.title("Distribution of protocols for Malign attacks", fontsize = 14)
plt.show()

# Corelation Matrix for the dataset
corelation_matrix = df.corr()
fig = plt.figure(figsize = (17, 17))
mask = np.zeros_like(corelation_matrix, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_theme(style = "darkgrid")
ax = sns.heatmap(corelation_matrix, square = True, annot = True, center = 0, vmin = -1, linewidths = .5, annot_kws = {"size" : 11}, mask = mask)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
plt.show()

#Encoding Categorical Features
print("Features which need to be encoded are : \n", categorical_features)

df = pd.get_dummies(df, columns = categorical_features, drop_first = True)

print("After Encoding DataFrame has {} rows and {} columns.".format(df.shape[0], df.shape[1]))

df.head(10)

df.info()

# Training and Testing
# Splitting Dependant (Label) and Independant Variables
x = df.drop(['label'], axis = 1)
y = df['label']

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ms = MinMaxScaler()
x = ms.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print(x_train.shape, x_test.shape)

accuracy_scores = []

print(accuracy_scores)

import pickle

# Machine Learning Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 1. K Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn_accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_scores.append(knn_accuracy * 100)
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

# Saving K Nearest Neighbors model
with open('models/knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

# 2. Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 1000)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
lr_accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_scores.append(lr_accuracy * 100)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# Saving Logistic Regression Model
with open('models/lr_model.pkl', 'wb') as file:
    pickle.dump(lr, file)

# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
dt_accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_scores.append(dt_accuracy * 100)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")

#Saving Decision Tree Model
with open('models/dt_model.pkl', 'wb') as file:
    pickle.dump(dt, file)

# 4. Support Vector Machine
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
svm_accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_scores.append(svm_accuracy * 100)
print(f"Support Vector Machine Accuracy: {svm_accuracy * 100:.2f}%")

# Saving Support Vector Machine model
with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump(svm, file)

# Defining Deep Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

model = keras.Sequential()
model.add(Dense(28, input_shape = (56, ), activation = "relu", name = "Hidden_Layer_1"))
model.add(Dense(10, activation = "relu", name = "Hidden_Layer_2"))
model.add(Dense(1, activation = "sigmoid", name = "Output_Layer"))
opt = keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ['accuracy'])
model.summary()

# Train Deep Neural Network for 50 epochs
dnn_model = model.fit(
    x_train,
    y_train,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    callbacks = None,
    validation_data = (x_test, y_test),
    shuffle = True,
    class_weight = None,
    sample_weight = None,
    initial_epoch = 0
)

# Accuracy and Loss for DNN Model
dnn_loss, dnn_accuracy = model.evaluate(x_test, y_test)
print(f"Deep Neural Network Accuracy: {dnn_accuracy * 100:.2f}%")

# Saving the DNN Model
model.save("models/DNN_model.keras")

accuracy_scores.append(dnn_accuracy * 100)

# Accuracy vs Number of Epochs
plt.figure(figsize = (10, 6))

accuracy = dnn_model.history['accuracy']
val_accuracy = dnn_model.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, color = "blue", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, color = "red", label = "Validation Accuracy")
plt.title("Accuracy vs Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss vs Number of Epochs
plt.figure(figsize = (10, 6))

loss = dnn_model.history['loss']
val_loss = dnn_model.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, loss, color = "blue", label = "Training loss")
plt.plot(epochs, val_loss, color = "red", label = "Validation loss")
plt.title("Loss vs Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print(accuracy_scores)

# Model Accuracies Comparison barplot
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'SVM', 'DNN']
model_names.reverse()
accuracy_scores.sort(reverse = True)
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))

sns.barplot(x=model_names, y=accuracy_scores, palette="muted", hue=model_names)

plt.legend(loc='upper right')

for i, accuracy in enumerate(accuracy_scores):
    plt.text(i, accuracy, f'{accuracy:.2f}%', ha='center', va='bottom')

plt.title("Model Accuracies")
plt.xlabel("\nModels")
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f5f5f5')
plt.tight_layout()
plt.show()