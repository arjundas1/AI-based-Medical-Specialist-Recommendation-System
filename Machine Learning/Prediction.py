import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

f = open('Symptom.csv')
arr = list()
csv_f = csv.reader(f)
for row in csv_f:
    arr.append(row[0])
f.close()
arr.remove('Disease')
for i in range(len(arr)):
    print(arr[i])

medical = list()
for i in range(len(arr)):
    if arr[i] == 'Fungal infection' or arr[i] == 'Acne' or arr[i] == 'Psoriasis' or arr[i] == 'Impetigo':
        medical[i] = 'Dermatologist'
    elif arr[i] == 'Allergy' or arr[i] == 'Drug Reaction':
        medical[i] = 'Allergist'
    elif arr[i] == 'GERD' or arr[i] == 'Peptic ulcer diseae' or arr[i] == 'Gastroenteritis' or arr[i] == 'Jaundice':
        medical[i] = 'Gastroenterologist'
    elif arr[i] == 'Chronic cholestasis' or arr[i] == 'hepatitis A' or arr[i] == 'hepatitis B' or \
            arr[i] == 'hepatitis C' or arr[i] == 'hepatitis D' or arr[i] == 'hepatitis E' or \
            arr[i] == 'Alcoholic hepatitis':
        medical[i] = 'Hepatologist'
    elif arr[i] == 'AIDS' or arr[i] == 'Chicken pox' or arr[i] == 'Common Cold':
        medical[i] = 'Primary Care Provider'
    elif arr[i] == 'Diabetes' or arr[i] == 'Hypothyroidism' or arr[i] == 'Hyperthyroidism' or arr[i] == 'Hypoglycemia':
        medical[i] = 'Endocrinologist'
    elif arr[i] == 'Bronchial Asthma' or arr[i] == 'Pneumonia':
        medical[i] = 'Pulmonologist'
    elif arr[i] == 'Hypertension' or arr[i] == 'Heart attack':
        medical[i] = 'Cardiologist'
    elif arr[i] == 'Migraine' or arr[i] == '(vertigo) Paroymsal  Positional Vertigo':
        medical[i] = 'Neurologist'
    elif arr[i] == 'Cervical spondylosis' or arr[i] == 'Osteoarthristis' or arr[i] == 'Arthritis':
        medical[i] = 'Orthopedic'
    elif arr[i] == 'Paralysis (brain hemorrhage)':
        medical[i] = 'Neurosurgeon'
    elif arr[i] == 'Malaria' or arr[i] == 'Dengue' or arr[i] == 'Typhoid' or arr[i] == 'Tuberculosis':
        medical[i] = 'Infectious Disease Doctor'
    elif arr[i] == 'Dimorphic hemmorhoids(piles)':
        medical[i] = 'Proctologist'
    elif arr[i] == 'Varicose veins':
        medical[i] = 'Vascular Surgeon'
    elif arr[i] == 'Urinary tract infection':
        medical[i] = 'Urologist'
    
df1 = pd.read_csv('Symptom.csv')
df2 = pd.read_csv('Symptom Description.csv')
df3 = pd.read_csv('Symptom Precaution.csv')
df4 = pd.read_csv('Symptom Severity.csv')

combined_df = pd.merge(df1, df2, on = 'Disease')
combined_df = pd.merge(combined_df , df3, on = 'Disease')

x = combined_df[['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4','Symptom_5']]
le = LabelEncoder()
for i in x.columns:
    x[i] = le.fit_transform(x[i].astype(str))
y = combined_df['Disease']
