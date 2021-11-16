import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from tkinter import *
from tkinter import messagebox
import sys 
import urllib
import urllib.request

df = pd.read_csv('Symptom.csv')
df1 = pd.read_csv('Symptom Severity.csv')
df4 = pd.read_csv('Symptom Specialist.csv')

specialist = df4['Specialist'].tolist()
specialist

edd = df4['Disease'].tolist()
edd

root = Tk()
root.title("Specialist Reccomendation Tool")
root.configure(bg='#ADD8E6')


Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

w2 = Label(root, justify=CENTER, text=" Specialist Reccomendation Tool ",bg='#ADD8E6')
w2.config(font=("Bookman Old Style", 25))
w2.grid(row=1, column=0, columnspan=2, padx=100)

NameLb1 = Label(root, text="", bg='#ADD8E6')
NameLb1.config(font=("Bookman Old Style", 20))
NameLb1.grid(row=5, column=1, pady=10,  sticky=W)

S1Lb = Label(root,  text="Symptom 1", bg='#ADD8E6')
S1Lb.config(font=("Bookman Old Style", 15))
S1Lb.grid(row=7, column=1, pady=10 , sticky=W)

S2Lb = Label(root,  text="Symptom 2", bg='#ADD8E6')
S2Lb.config(font=("Bookman Old Style", 15))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root,  text="Symptom 3", bg='#ADD8E6')
S3Lb.config(font=("Bookman Old Style", 15))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root,  text="Symptom 4", bg='#ADD8E6')
S4Lb.config(font=("Bookman Old Style", 15))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root,  text="Symptom 5", bg='#ADD8E6')
S5Lb.config(font=("Bookman Old Style", 15))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)


lr = Button(root, text="Predict",height=2, width=20, command=message)
lr.config(font=("Bookman Old Style", 15))
lr.grid(row=15, column=1,pady=10)

#OPTIONS = sorted(symptoms)
OPTIONS = df1['Symptom']


S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

NameLb = Label(root, text="", bg='#ADD8E6')
NameLb.config(font=("Bookman Old Style", 20))
NameLb.grid(row=13, column=1, pady=10,  sticky=W)

NameLb = Label(root, text="", bg='#ADD8E6')
NameLb.config(font=("Bookman Old Style", 15))
NameLb.grid(row=17, column=1, pady=10,  sticky=W)

#t3 = Text(root, height=2, width=20)
#t3.config(font=("Bookman Old Style", 20))
#t3.grid(row=18, column=1 , padx=10)

t4 = Text(root, height=2, width=20)
t4.config(font=("Bookman Old Style", 20))
t4.grid(row=20, column=1 , padx=10)

root.mainloop()

def message():
    if (Symptom1.get() == "None" and  Symptom2.get() == "None" and Symptom3.get() == "None" and Symptom4.get() == "None" and Symptom5.get() == "None"):
        messagebox.showinfo("OPPS!!", "ENTER  SYMPTOMS PLEASE")
    else :
        SVM()

def SVM():
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
    psy = [psymptoms + nulls]

    prob = model.predict(psy)
    #t3.delete("1.0", END)
    #t3.insert(END, prob[0])

    print(prob[0])
    
    spec= prob[0]
    for l in range(40):
        if spec==edd[l]:
            t4.delete("1.0", END)
            t4.insert(END, specialist[l])

   