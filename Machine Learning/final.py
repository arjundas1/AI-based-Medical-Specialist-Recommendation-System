import tkinter as tk
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

class mainClass(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome to the AI-based Medical Specialist Recommendation System", bg='LightSteelBlue1')
        label.config(font=("Bookman Old Style", 28))
        label.pack(pady=20, padx=20)
        self.configure(bg='LightSteelBlue1')
        NameLb = Label(self, text="If you are suffering from a disease and you do not have an idea as to which doctor "
                                  "to go to, so as to get started with the correct treatment, then this recommendation "
                                  "system can help you advise the kind of doctor you must visit, based on the symptoms "
                                  "that your body is showing.", wraplength=1350, bg='LightSteelBlue1')
        NameLb.config(font=("Bookman Old Style", 16))
        NameLb.pack(pady=20, padx=40)

        NameLb2 = Label(self, text="Steps to get an authentic recommendation:                                          "
                                   "                 \n1. Click on the 'Get Recommendation' button. You will be "
                                   "redirected to a new page.\n2.Choose the five most prevalent disease symptoms that "
                                   "your body is showing.      \n\nPro-tip: Assess yourself thoroughly before entering "
                                   "the details in order to get the most apt recommendation from our system.",
                        bg='khaki2')
        NameLb2.config(font=("Bookman Old Style", 16))
        NameLb2.pack(pady=40, padx=40)

        button = tk.Button(self, text="Get Recommendation", command=lambda: controller.show_frame(PageOne))
        button.config(font=("Bookman Old Style", 18))
        button.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='RosyBrown2')
        label = tk.Label(self, text="Enter the five most prevalent symptoms that you are facing at the moment", bg='RosyBrown2')
        label.config(font=("Bookman Old Style", 20))
        label.pack(pady=2, padx=2)

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

        OPTIONS = df1['Symptom']

        S1Lb = Label(self, text="Symptom 1", bg='RosyBrown2')
        S1Lb.config(font=("Bookman Old Style", 25))
        S1Lb.pack(padx=5, pady=5)

        S1En = OptionMenu(self, Symptom1, *OPTIONS)
        S1En.config(font=("Bookman Old Style", 20))
        S1En.pack(padx=5, pady=5)

        S2Lb = Label(self, text="Symptom 2", bg='RosyBrown2')
        S2Lb.config(font=("Bookman Old Style", 25))
        S2Lb.pack(padx=5, pady=5)

        S2En = OptionMenu(self, Symptom2, *OPTIONS)
        S2En.config(font=("Bookman Old Style", 20))
        S2En.pack(padx=5, pady=5)

        S3Lb = Label(self, text="Symptom 3", bg='RosyBrown2')
        S3Lb.config(font=("Bookman Old Style", 25))
        S3Lb.pack(padx=5, pady=5)

        S3En = OptionMenu(self, Symptom3, *OPTIONS)
        S3En.config(font=("Bookman Old Style", 20))
        S3En.pack(padx=5, pady=5)

        S4Lb = Label(self, text="Symptom 4", bg='RosyBrown2')
        S4Lb.config(font=("Bookman Old Style", 25))
        S4Lb.pack(padx=5, pady=5)

        S4En = OptionMenu(self, Symptom4, *OPTIONS)
        S4En.config(font=("Bookman Old Style", 20))
        S4En.pack(padx=5, pady=5)

        S5Lb = Label(self, text="Symptom 5", bg='RosyBrown2')
        S5Lb.config(font=("Bookman Old Style", 25))
        S5Lb.pack(padx=5, pady=5)

        S5En = OptionMenu(self, Symptom5, *OPTIONS)
        S5En.config(font=("Bookman Old Style", 20))
        S5En.pack(padx=5, pady=5)

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
                    t4.insert(END, f"You are recommended to visit any {specialist[l]}")

        def message():
            count = 0
            if Symptom1.get() == "None":
                count += 1
            if Symptom2.get() == "None":
                count += 1
            if Symptom3.get() == "None":
                count += 1
            if Symptom4.get() == "None":
                count += 1
            if Symptom5.get() == "None":
                count += 1
            if count != 0:
                messagebox.showinfo("Warning", "Please enter all 5 symptoms")
            arr = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
            flag = True
            for i in range(len(arr)):
                for j in range(i):
                    if arr[i] == arr[j]:
                        flag = False
                        break
            if flag != True:
                messagebox.showinfo("Warning", "Same symptoms cannot be repeated")
            else:
                RF()

        lr = Button(self, text="Predict", height=1, width=12, command=message)
        lr.config(font=("Bookman Old Style", 22))
        lr.pack(padx=5, pady=5)

        t4 = Text(self, height=2, width=40)
        t4.config(font=("Bookman Old Style", 20))
        t4.pack(padx=5, pady=5)


app = mainClass()
app.mainloop()
