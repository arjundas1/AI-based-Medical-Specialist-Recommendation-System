import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from tkinter import messagebox

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

df2 = pd.read_csv('Disease Specialist.csv')
specialist = df2['Specialist'].tolist()
edd = df2['Disease'].tolist()

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

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(data, labels)

root = Tk()
root.title("Specialist Recommendation Tool")
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

w2 = Label(root, justify=CENTER, text=" Specialist Recommendation Tool ", bg='#ADD8E6')
w2.config(font=("Bookman Old Style", 25))
w2.grid(row=1, column=0, columnspan=2, padx=100)

NameLb1 = Label(root, text="", bg='#ADD8E6')
NameLb1.config(font=("Bookman Old Style", 20))
NameLb1.grid(row=5, column=1, pady=10, sticky=W)

S1Lb = Label(root, text="Symptom 1", bg='#ADD8E6')
S1Lb.config(font=("Bookman Old Style", 15))
S1Lb.grid(row=7, column=1, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", bg='#ADD8E6')
S2Lb.config(font=("Bookman Old Style", 15))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", bg='#ADD8E6')
S3Lb.config(font=("Bookman Old Style", 15))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", bg='#ADD8E6')
S4Lb.config(font=("Bookman Old Style", 15))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", bg='#ADD8E6')
S5Lb.config(font=("Bookman Old Style", 15))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)


def RF():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    psy = [psymptoms]
    prob = rf.predict(psy)
    print(prob[0])
    spec = prob[0]
    for l in range(40):
        if spec == edd[l]:
            t4.delete("1.0", END)
            t4.insert(END, specialist[l])


def message():
    if (Symptom1.get() == "None" and Symptom2.get() == "None" and Symptom3.get() == "None" and Symptom4.get() == "None"
            and Symptom5.get() == "None"):
        messagebox.showinfo("OPPS!!", "ENTER  SYMPTOMS PLEASE")
    else:
        RF()


lr = Button(root, text="Predict", height=2, width=20, command=message)
lr.config(font=("Bookman Old Style", 15))
lr.grid(row=15, column=1, pady=10)

OPTIONS = df1['Symptom']

S1En = OptionMenu(root, Symptom1, *OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2, *OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3, *OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4, *OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5, *OPTIONS)
S5En.grid(row=11, column=1)

NameLb = Label(root, text="", bg='#ADD8E6')
NameLb.config(font=("Bookman Old Style", 20))
NameLb.grid(row=13, column=1, pady=10, sticky=W)

NameLb = Label(root, text="", bg='#ADD8E6')
NameLb.config(font=("Bookman Old Style", 15))
NameLb.grid(row=17, column=1, pady=10, sticky=W)

t4 = Text(root, height=2, width=20)
t4.config(font=("Bookman Old Style", 20))
t4.grid(row=20, column=1, padx=10)

root.mainloop()
