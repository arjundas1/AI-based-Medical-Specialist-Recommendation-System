import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

df = pd.read_csv('Symptom.csv')
df.drop(['Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13',
         'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17'], axis=1, inplace=True)
cols = df.columns
data = df[cols].values.flatten()
s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)
vals = df.values

df1 = pd.read_csv('Symptom Severity.csv')
symptoms = df1['Symptom'].unique()
for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination', 0)
df = d.replace('foul_smell_of urine', 0)
data = df.iloc[:, 1:].values
labels = df['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
lr = LogisticRegression(solver="newton-cg", max_iter=3000)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print(round(lracc*100, 3), "%", sep="")

x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size=0.85)
model = SVC()
model.fit(x_train, y_train)
preds = model.predict(x_test)
acc = metrics.accuracy_score(y_test, preds)
print(round(acc * 100, 3), "%", sep="")

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(trainX, trainY)
y_pred = classifier.predict(testX)
knnacc = classifier.score(testX, testY)
print(round(knnacc*100, 3), "%", sep="")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=9)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
preds = rf.predict(x_test)
print(round(rf.score(x_test, y_test) * 100, 3), "%", sep="")

conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100)
sns.heatmap(df_cm, cmap="RdPu")
plt.show()
