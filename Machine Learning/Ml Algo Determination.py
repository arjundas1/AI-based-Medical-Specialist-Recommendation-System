import pandas as pd
import sklearn
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df1 = pd.read_csv('C:/Users/Arjun/Downloads/Dataset/Symptom.csv')
df2 = pd.read_csv('C:/Users/Arjun/Downloads/Dataset/Symptom Description.csv')
df3 = pd.read_csv('C:/Users/Arjun/Downloads/Dataset/Symptom Precaution.csv')
df4 = pd.read_csv('C:/Users/Arjun/Downloads/Dataset/Symptom Severity.csv')

combined_df = pd.merge(df1, df2, on = 'Disease')
combined_df = pd.merge(combined_df , df3, on = 'Disease')

x = combined_df[['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4','Symptom_5']]
le = LabelEncoder()
for i in x.columns:
    x[i] = le.fit_transform(x[i].astype(str))
y = combined_df['Disease']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2,random_state=10)
lr = LogisticRegression(solver="newton-cg",max_iter=3000)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print(round(lracc*100, 3), "%", sep="")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=10)
svmmodel = svm.SVC(kernel="linear", C=2)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print(round(svmacc*100, 3), "%", sep="")

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state = 42)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(trainX, trainY)
y_pred = classifier.predict(testX)
knnacc = classifier.score(testX, testY)
print(round(knnacc*100, 3),"%", sep="")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=9)
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print(round(rf.score(x_test,y_test) * 100, 3),"%",sep="")