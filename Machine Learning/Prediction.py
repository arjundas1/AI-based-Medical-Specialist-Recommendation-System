import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=9)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =',
      accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm, cmap="RdPu")
plt.show()

from sklearn import tree
fig = plt.figure(figsize=(25, 20))
treeviz = tree.plot_tree(rf.estimators_[0],
                         filled=True, impurity=True,
                         rounded=True,
                         feature_names=data,
                         class_names=labels)
plt.show()

def doctor(prob):
    if prob == 'Fungal infection' or prob == 'Acne' or prob == 'Psoriasis' or prob == 'Impetigo':
        return 'Dermatologist'
    elif prob == 'Allergy' or prob == 'Drug Reaction':
        return 'Allergist'
    elif prob == 'GERD' or prob == 'Peptic ulcer diseae' or prob == 'Gastroenteritis' or prob == 'Jaundice':
        return 'Gastroenterologist'
    elif prob == 'Chronic cholestasis' or prob == 'hepatitis A' or prob == 'hepatitis B' or \
            prob == 'hepatitis C' or prob == 'hepatitis D' or prob == 'hepatitis E' or \
            prob == 'Alcoholic hepatitis':
        return 'Hepatologist'
    elif prob == 'AIDS' or prob == 'Chicken pox' or prob == 'Common Cold':
        return 'Primary Care Provider'
    elif prob == 'Diabetes' or prob == 'Hypothyroidism' or prob == 'Hyperthyroidism' or prob == 'Hypoglycemia':
        return 'Endocrinologist'
    elif prob == 'Bronchial Asthma' or prob == 'Pneumonia':
        return 'Pulmonologist'
    elif prob == 'Hypertension' or prob == 'Heart attack':
        return 'Cardiologist'
    elif prob == 'Migraine' or prob == '(vertigo) Paroymsal  Positional Vertigo':
        return 'Neurologist'
    elif prob == 'Cervical spondylosis' or prob == 'Osteoarthristis' or prob == 'Arthritis':
        return 'Orthopedic'
    elif prob == 'Paralysis (brain hemorrhage)':
        return 'Neurosurgeon'
    elif prob == 'Malaria' or prob == 'Dengue' or prob == 'Typhoid' or prob == 'Tuberculosis':
        return 'Infectious Disease Doctor'
    elif prob == 'Dimorphic hemmorhoids(piles)':
        return 'Proctologist'
    elif prob == 'Varicose veins':
        return 'Vascular Surgeon'
    elif prob == 'Urinary tract infection':
        return 'Urologist'


def predd(S1, S2, S3, S4, S5):
    psymptoms = [S1, S2, S3, S4, S5]
    print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    psy = [psymptoms]
    pred2 = rf.predict(psy)
    print(pred2[0])
    print(f'You are advised to visit any {doctor(pred2[0])}')


sympList = df1["Symptom"].to_list()
'''
for i in range(len(sympList)):
    print(sympList[i])
'''
predd(sympList[7], sympList[2], sympList[5], sympList[1], sympList[8])
