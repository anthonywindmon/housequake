#Anthony Windmon - Several classifiers tested on a bank dataset.
#Includes implementation of SMOTE, Random Oversampling and Random Undersampling
#References - YouTube, Stack Overflow, Python's website, Quora, TowardsDataScience.com, etc.

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#loads bank dataset
df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\Projects\\bank-additional-full.csv')

#PRE-PROCESSING DATA
#changing 'poutcome' to numbers
def trans_poutcome(x):
    if x == 'success':
        return 1
    if x == 'failure':
        return 0
    else:
        return 2

#changing 'day_of_week' to numbers
def trans_dayOfWeek(x):
    if x == 'mon':
        return 1
    if x == 'tue':
        return 2
    if x == 'wed':
        return 3
    if x == 'thu':
        return 4
    if x == 'fri':
        return 5
    if x == 'sat':
        return 6
    else:
        return 7

def trans_month(x):
    if x == 'jan':
        return 1
    if x == 'feb':
        return 2
    if x == 'mar':
        return 3
    if x == 'apr':
        return 4
    if x == 'may':
        return 5
    if x == 'jun':
        return 6
    if x == 'jul':
        return 7
    if x == 'aug':
        return 8
    if x == 'sep':
        return 9
    if x == 'oct':
        return 10
    if x == 'nov':
        return 11
    else:
        return 12

def trans_job(x):
    if x == 'admin':
        return 1
    if x == 'blue-collar':
        return 2
    if x == 'entrepreneur':
        return 3
    if x == 'housemaid':
        return 4
    if x == 'management':
        return 5
    if x == 'retired':
        return 6
    if x == 'self-employed':
        return 7
    if x == 'services':
        return 8
    if x == 'student':
        return 9
    if x == 'technician':
        return 10
    if x == 'unemployed':
        return 11
    else:
        return 12

def trans_marital(x):
    if x == 'divorced':
        return 1
    if x == 'married':
        return 2
    if x == 'single':
        return 3
    else:
        return 4

def trans_education(x):
    if x == 'basic.4y':
        return 1
    if x == 'basic.6y':
        return 2
    if x == 'basic.9y':
        return 3
    if x == 'high.school':
        return 4
    if x == 'illiterate':
        return 5
    if x == 'professional.course':
        return 6
    if x == 'university.degree':
        return 7
    else:
        return 8

def trans_default(x):
    if x == 'yes':
        return 1
    if x == 'no':
        return 2
    else:
        return 3

def trans_housingLoan(x):
    if x == 'yes':
        return 1
    if x == 'no':
        return 2
    else:
        return 3

def trans_personalLoan(x):
    if x == 'yes':
        return 1
    if x == 'no':
        return 2
    else:
        return 3

def trans_contact(x):
    if x == 'cellular':
        return 1
    if x == 'telephone':
        return 2

#replacing nominal data with numerical
df['trans_poutcome'] = df['poutcome'].apply(trans_poutcome)
df['trans_dayOfWeek'] = df['day_of_week'].apply(trans_dayOfWeek)
df['trans_month'] = df['month'].apply(trans_month)
df['trans_job'] = df['job'].apply(trans_job)
df['trans_marital'] = df['marital'].apply(trans_marital)
df['trans_education'] = df['education'].apply(trans_education)
df['trans_default'] = df['default'].apply(trans_default)
df['trans_housingLoan'] = df['housing'].apply(trans_housingLoan)
df['trans_personalLoan'] = df['loan'].apply(trans_personalLoan)
df['trans_contact'] = df['contact'].apply(trans_contact)

#droping unwanted columns
df = df.drop(columns=['poutcome','day_of_week','month','job','marital',
            'education','default', 'housing','loan','contact'])
#prints first 6 elements of dataset
print(df.head())
shape = df.shape
print('shape of data =', shape)
target = df['y'] #the class colunn is called 'y', containing 'yes' and 'no' class names

feature_names = df.columns.tolist()
feature_names.remove('y')
print('feature names =', feature_names)

#checks to make sure data is balanced, but counting number of yes and no
yes_Count = 0
no_Count = 0
for class_label in target:
    if class_label == 'yes':
        yes_Count += 1
    else:
        no_Count += 1
#Proves that dataset is very unbalanced
print('Number of yes =', yes_Count)
print('Number of no =', no_Count)

#Splitting data into training and training
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['y']), target, test_size=0.15,
                                                        random_state=2)
print('--------------------------BEFORE SAMPLING-----------------------------------')
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

#RF Classification w/ unbalanced data
model = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model.fit(X_train, y_train) #training model
result = model.score(X_test, y_test) #calculate accuracy
print("Accuracy =", result)

print('------------------------SMOTE--------------------------')
#Balance data using SMOTE
sm = SMOTE(random_state = 12, ratio = 1.0, k_neighbors = 6, sampling_strategy='minority')
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train) #we only oversample the training data

print("The length of the training set (After SMOTE) =", len(X_train_smote)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

model_smote = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model_smote.fit(X_train_smote, y_train_smote)
results_smote = model_smote.score(X_test, y_test)
print("RF Accuracy (Before SMOTE) =", result)
print('RF Accuracy (After SMOTE) =', results_smote)

print('-------------------------------RANDOM OVERSAMPLING--------------------')
#Balance data with Random OverSampling
random_os = RandomOverSampler(random_state=12)
X_train_ros, y_train_ros = random_os.fit_resample(X_train, y_train)
print("The length of the training set (After ROS) =", len(X_train_ros)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

model_ros = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model_ros.fit(X_train_ros, y_train_ros)
results_ros = model_ros.score(X_test, y_test)
print("RF Accuracy (Before SMOTE) =", result)
print('RF Accuracy (After SMOTE) =', results_smote)
print('RF Accuracy (After ROS) =', results_ros)

print('---------------------------RANDOM UNDERSAMPLING----------------------')
#Balance data with Random UnderSampling
random_us = RandomUnderSampler(random_state=12)
X_train_rus, y_train_rus = random_us.fit_resample(X_train, y_train)
print("The length of the training set (After ROS) =", len(X_train_rus)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

model_rus = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model_rus.fit(X_train_rus, y_train_rus)
results_rus = model_rus.score(X_test, y_test)
print("RF Accuracy (Before SMOTE) =", result)
print('RF Accuracy (After SMOTE) =', results_smote)
print('RF Accuracy (After ROS) =', results_ros)
print('RF Accuracy (After RUS)', results_rus)

#Feature Selection
model_features = ExtraTreesClassifier(n_estimators=100)
model_features.fit(X_train, y_train)
print('Strongest Features =',model_features.feature_importances_) #feature_importances_ is a feature which gives us the strongest features
strongest_features = pd.Series(model_features.feature_importances_, index=X_train.columns)
strongest_features.nlargest(10).plot(kind='barh')
plt.xlabel('Scores')
plt.ylabel('Features')
plt.title('10 Strongest Features')
plt.show()

print('--------------------ACCURACY AFTER FEATURE SELECTION-----------------------')
results_feature_selection = model_features.score(X_test, y_test)
print('RF Accuracy (After Features Selection) =', results_feature_selection)

#Confusion Matrix
y_predicted = model.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("RF Confusion Matrix: \n", confuse_matrix)

#precision, recall & f-measure
classifier_report = classification_report(y_test,y_predicted)
print(classifier_report)

print('-----------------------DECISION TREE----------------------')
model_two = DecisionTreeClassifier()
model_two_smote = DecisionTreeClassifier()
model_two_ros = DecisionTreeClassifier()
model_two_rus = DecisionTreeClassifier()

model_two.fit(X_train, y_train)
model_two_smote.fit(X_train_smote, y_train_smote)
model_two_ros.fit(X_train_ros, y_train_ros)
model_two_rus.fit(X_train_rus, y_train_rus)

results_two = model_two.score(X_test, y_test)
results_two_smote = model_two_smote.score(X_test, y_test)
results_two_ros = model_two_ros.score(X_test, y_test)
results_two_rus = model_two_rus.score(X_test, y_test)

print('Decision Tree Accuracy =', results_two)
print('Decision Tree Accuracy (SMOTE)', results_two_smote)
print('Decision Tree Accuracy (ROS)', results_two_ros)
print('Decision Tree Accuracy (ROS)', results_two_rus)

print('----------------------LOGISTIC REGRESSION-------------------')
model_three = LogisticRegression()
model_three_smote = LogisticRegression()
model_three_ros = LogisticRegression()
model_three_rus = LogisticRegression()

model_three.fit(X_train, y_train)
model_three_smote.fit(X_train_smote, y_train_smote)
model_three_ros.fit(X_train_ros, y_train_ros)
model_three_rus.fit(X_train_rus, y_train_rus)

results_three = model_three.score(X_test, y_test)
results_three_smote = model_three_smote.score(X_test, y_test)
results_three_ros = model_three_ros.score(X_test, y_test)
results_three_rus = model_three_rus.score(X_test, y_test)

print('Logistic Reg. Accuracy =', results_three)
print('Logistic Reg. Accuracy (SMOTE) =', results_three_smote)
print('Logistic Reg. Accuracy (ROS) =', results_three_ros)
print('Logistic Reg. Accuracy (RUS) =', results_three_rus)

print('--------------------------------------------------')
#Cohen Kappa Score is a measure that works well with multiclass and class imbalances
#Tell us how good our classifier is compaired to other classifiers
y_predicted = model.predict(X_test)
kappa_score = cohen_kappa_score(y_test, y_predicted)
print('Cohen Kappa Score =', kappa_score)
