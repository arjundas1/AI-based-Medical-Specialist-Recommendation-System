<p align="center">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Cover2.jpeg" width="650" height="250">
  </a>
</p>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#project-team">Project Team</a></li>
    <li><a href="#project-objective">Project Objective</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Introduction

The applications of Artificial Intelligence in the field of Healthcare has seen rapid development in the present time. There is a lot of improvement scope that can be observed, and several implementations and upgrade can be brought to the existing AI technology. Collection of proper real-life data helps a lot in the implementation of efficient and accurate AI models. In the field of heathcare, accurate identification of diseases based on the patients' symptoms can be one of the first help that smart systems can provide to the doctors. However, due to very minor and negligible differences between quite a few diseases, along with the fact that new or unknown diseases can be found, predicting such diseases accurately can be problematic due to no availability of datasets. Therefore, Artificial Intelligence can still need to depend on actual doctors for identifying certain types of diseases. Keeping this fact and mind, we are generalising the aforementioned concept to create a system which recommends the user a specialised doctor based on the symptoms that they are faced by them.

## Project Team

Guidance Professor: Dr. Joshan Athanesious J, School of Computer Science and Engineering, VIT Vellore.

The project members are :-

|Sl.No. | Name  | Registration No. |
|-| ------------- |:-------------:|
|1|  Esha Jawaharlal     | 19BCE2459  |
|2|    Arjun Das         | 20BDS0129  |

## Project Objective

To build a recommendation system that uses classification Machine Learning algorithm to predict the disease of a patient based on the symptoms provided by the patient. The predicted disease will not be disclosed to the patient, instead a relevant medical specialist will be recommended to the patient. 

This system aims to help the patients who are confused about the disease they have contracted or do not have a basic idea as to which medical specialist should they visit in order get the right treatment. 

## Methodology

1. Finding a suitable dataset that is relevant to the aim of our project.
2. Determining the usability of the dataset, cleaning the data (if required) and preprocessing the data.
3. Visualizing the dataset for better understanding of the data present in it.
4. Finding an apt Machine Learning algorithm that yields a great prediction accuracy.
5. Working on the overfitting of the model (in case it is overfitted).
6. Building a user interface that interacts with the patients in the front end.
7. Connecting the ML model in the back end to the front end.

## Tools Used

Python Language Libraries:
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - Scikitlearn
  - Tkinter

## Implementation

### Dataset

The dataset that will be used for this project is a publicly available dataset that was found on Kaggle, a Machine Learning and Data Science Community platform. It is created by Pranay Patil and Pratik Rathod. The dataset can be found in the [Dataset](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/tree/main/Dataset) folder of this respository. It is a very new dataset as it was uploaded in 2019 and updated in 2020. The usability of this dataset is rated very high by Kaggle (9.7/10). There are four csv files in the Dataset folder:

- [Symptom.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom.csv): This is the most important csv file, where the symptoms and their corresponding disease has been mentioned. The dataset consists of 17 columns of symptoms and 1 column for Disease name. The symptom columns (named as Symptom_1 to Symptom_17) has a lot of null values in their cells, indicating to the fact that a disease can have less than 17 symptoms.
- [Symptom Description.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Description.csv): This csv file contains the description of every unique disease that has been mentioned in the Symptom.csv file. Although the name of the file is Symptom description, it does not have any description of any of the symptom.
- [Symptom Severity.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Severity.csv): This csv file contains the level of severity of the symptoms that are present in Symptom.csv. The highest (indicating to most critical severity) symptom has been given 7 and the lease (indicating to least critical severity) has been given 1.
- [Symptom Precaution.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Precaution.csv): This csv file contains 4 columns of simple precautions that can be taken to avoid contracting the list of unique diseases present in Symptom.csv.

### Reading the data

We read the dataset using Python's pandas library. We only require Symptom.csv and Symptom severity.csv for building the ML Algorithm.
```python
import pandas as pd
df = pd.read_csv('Symptom.csv')
df1 = pd.read_csv('Symptom Description.csv')
df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Disease</th>
      <th>Symptom_1</th>
      <th>Symptom_2</th>
      <th>Symptom_3</th>
      <th>Symptom_4</th>
      <th>Symptom_5</th>
      <th>Symptom_6</th>
      <th>Symptom_7</th>
      <th>Symptom_8</th>
      <th>Symptom_9</th>
      <th>Symptom_10</th>
      <th>Symptom_11</th>
      <th>Symptom_12</th>
      <th>Symptom_13</th>
      <th>Symptom_14</th>
      <th>Symptom_15</th>
      <th>Symptom_16</th>
      <th>Symptom_17</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic_patches</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Fungal infection</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic_patches</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic_patches</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>dischromic_patches</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

### Data visualization

Once the csv files are read, we can use Python visualization libraries such as Matplotlib and Seaborn to visualise the data in better, user friendly way.

1. Importing libraries
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

 2. Visualising the count of various severity levels of symptoms using a histogram
```python
vis1 = plt.hist(df1["weight"], color=sns.color_palette()[8])
plt.xlabel("Count")
plt.ylabel("Severity")
plt.show()
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Severity-count%20histogram.png" width="350" height="250">
  </a>
</p>
From this figure, we infer that most of the symptoms mentioned in this datset has severity 
levels 4 and 5. This helps us realize the kind of symptoms available in the dataset and 
whether we wish to weigh the in the severity for each symptom during data preprocessing 
or not.

3. In order to understand the various symptoms that are present in the columns, we can make pie charts of every to see the variety of symptoms present in each column. For easy understanding, we have included a pie plot of 'Symptom_12' column
```python
symptom = df["Symptom_12"].value_counts()
vis2 = plt.pie(symptom, labels=symptom.index, startangle=100, counterclock=False)
plt.title("Symptom 12")
plt.show()
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Pie%20plot%20Symptom%2012.png" width="500" height="300">
  </a>
</p>
The variety of symptoms are highly different in various columns. As we move towards 
the right columns, the number of null values increase and hence the variety decreases 
altogether. The difference in the number of symptoms in 'Symptom_14' column as a ring 
plot against that of 'Symptom_12' shown above.

```python
col = ['greenyellow', 'orchid', 'burlywood', 'salmon']
symptom = df["Symptom_14"].value_counts()
vis3 = plt.pie(symptom, labels=symptom.index, startangle=90,
               counterclock=False, wedgeprops={'width': 0.4}, colors=col)
plt.title("Symptom 14")
plt.show()
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Ring%20plot%20Symptom%2014.png" width="360" height="260">
  </a>
</p>
The 11 variety of symptoms reduced to only 4 in Symptom_14 column. This enables us 
to decide the number of symptom columns that must be taken for consideration while 
creating the model. 

### Data Preprocessing
1. From the above head of df1 dataframe, we observe that a lot of Null values are present in the dataset. Therefore we count the total number of NaN values in the dataset.
```python
df.isna().count()
```
```
Disease          0
Symptom_1        0
Symptom_2        0
Symptom_3        0
Symptom_4      348
Symptom_5     1206
Symptom_6     1986
Symptom_7     2652
Symptom_8     2976
Symptom_9     3228
Symptom_10    3408
Symptom_11    3726
Symptom_12    4176
Symptom_13    4416
Symptom_14    4614
Symptom_15    4680
Symptom_16    4728
Symptom_17    4848
dtype: int64
```

2. Due to the large number of null values from Sypmtom_6 onwards, we will not be using 
the columns from Symptom_6 onwards in our Machine Learning model. Therefore, we 
drop those columns from the data frame that we have created.
```python
df.drop(['Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13',
         'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17'], axis=1, inplace=True)
```

3. The data requires to be cleaned. The column values need to be flattened to a single 
dimension and reshaped. Also, the null values need to be addressed and filled up with an 
appropriate value. Here, we fill the null values with 0.

```python
cols = df.columns
data = df[cols].values.flatten()
s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)
vals = df.values
```

4.  We match the data frame of Symptom.csv and their weights in Symptom Severity.csv for 
building a precise model. We also manually replace the weights with 0 for those symptoms 
that are not present in Symptom Severity.csv but are present in Symptom.csv.
```python
symptoms = df1['Symptom'].unique()
for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination', 0)
df = d.replace('foul_smell_of urine', 0)
```

5. We create separate x and y axis data columns wherein the column of interest is retained in y(here, labels) and the main prediction data in x(here, data)
```python
data = df.iloc[:, 1:].values
labels = df['Disease'].values
```

### ML Model

To find out the suitable machine learning algorithm, we have implemented several algorithms 
to determine the one yielding highest prediction accuracy. The conditions for train-test split 
have also been varied according to our understanding of effiency yield. However, every model 
uses 20% data as testing set.
```python
from sklearn.model_selection import train_test_split
```
#### Logistic Regression

Although the name has regression in it, Logistic regression is a classification algorithm. We identified the most efficient solver for our dataset to be newton-cg. Since covergence passes at 3000 iterations, the algorithm is slow with not a high prediction accuracy score.
```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="newton-cg", max_iter=3000)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print(round(lracc*100, 3), "%", sep="")
```
```
79.167%
```
We do not wish to choose Logistic Regression as the preferred ML Algortihm for our model

#### Support Vector Machine

SVM gives a very high prediction accuracy when implemented. However, this algorithm is very slow. This is possibly because of the eager learning construct of this algorithm, or that finding a suitable hyperplane is time consuming.
```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, shuffle=True)
from sklearn import svm, metrics
svmmodel = svm.SVC(kernel="linear", C=2)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print(round(svmacc*100, 3), "%", sep="")
```
```
91.057%
```
We do not wish to implement a time consuming algorithm like SVM, therefore we try other algorithms

#### K-Nearest Neighbors

KNN is a lazy learning algorithm, that yield very high prediction accuracy on the testing dataset. This is a very suitable algorithm for our purpose.
```python
trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.2, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(trainX, trainY)
y_pred = classifier.predict(testX)
knnacc = classifier.score(testX, testY)
print(round(knnacc*100, 3), "%", sep="")
```
```
93.801%
```
Although this algorithm yields a very high prediction accuracy, we would try one last algorithm to see if we get higher prediction accuracy

#### Random Forest

Random Forest is based on Decision trees concepts where a number of Decision Trees are implemented and the best one is used. We have set number of estimators at 100.
```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=9)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print(round(rf.score(x_test,y_test) * 100, 3),"%",sep="")
```
```
94.634%
```
Since we are getting the highest prediction accuracy without any overfitting, we decide to use Random Forest further in this project.

F1 Score and visual representation(Heatmap) of the confusion matrix:
```python
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100)
sns.heatmap(df_cm, cmap="RdPu")
plt.show()
```
```
F1-score% = 94.10676945338828
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Heatmap.png" width="500" height="450">
  </a>
</p>

Visualisation of the kind of decision tree formed:
```python
from sklearn import tree
fig = plt.figure(figsize=(25, 20))
treeviz = tree.plot_tree(rf.estimators_[0],
                         filled=True, impurity=True,
                         rounded=True,
                         feature_names=data,
                         class_names=labels)
```
<p align="center">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Decision%20Tree.jpeg" width="650" height="500">
  </a>
</p>

### User Interface

The patients can get a hint on which specialist to visit based on the symptoms they have 
provided. The user interface is designed using the tkinter library present in python. Tkinter is 
a python binding to the Tk GUI toolkit and is python’s de facto standard GUI. 

The user interface created is a Recommendation System, that would take five symptoms in the 
form of an input and display the specialist to be visited. The GUI created consists of a home 
page, where the guidance to give the inputs have been mentioned, and a prediction page where 
inputs are taken and prediction is generated. 

Since, no symptom option can be left blank and the same symptom cannot be repeated in the 
five required symptoms, users get to see warning message boxes when the recommendation 
system rules are not abided. 
```python
import tkinter as tk
from tkinter import *

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
```

![image](https://user-images.githubusercontent.com/72820515/150387908-472c7a8d-9e40-4c88-ba58-8f87a7c44de3.png)

![image](https://user-images.githubusercontent.com/72820515/150388101-3f9401eb-1292-4068-958d-2cdab3b2d3fe.png)


## References
- [_Baclic, O., Tunis, M., Young, K., Doan, C., Swerdfeger, H., & Schonfeld, J. (2020). Artificial intelligence in public health: Challenges and opportunities for public health made possible by advances in natural language processing. Canada Communicable Disease Report, 46(6), 161._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/Challenges%20and%20opportunities%20for%20public%20health.pdf)
- [_Harsh, M., Suhas, D., Manthan, T., Anas, D. (2021). AI Based Healthcare Chatbot System by Using Natural Language. International Journal of Scientific Research and Engineering Development, Volume 4 Issue 2._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/AI%20Based%20Healthcare%20Chatbot%20System%20by%20Using%20Natural%20Language.pdf)
- [_Khanna, A., Pandey, B., Vashishta, K., Kalia, K., Pradeepkumar, B., & Das, T. (2015). A study of today’s ai through chatbots and rediscovery of machine intelligence. International Journal of u-and e-Service, Science and Technology, 8(7), 277-284._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/A%20Study%20of%20Today's%20A.I.%20through%20Chatbots%20and%20Rediscovery%20of%20Machine%20Intelligence.pdf)
- [_Palanica, A., Flaschner, P., Thommandram, A., Li, M., & Fossat, Y. (2019). Physicians’ perceptions of chatbots in health care: cross-sectional web-based survey. Journal of medical Internet research, 21(4), e12887._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/Physicians%E2%80%99%20Perceptions%20of%20Chatbots%20in%20Health%20Care%20Cross-Sectional%20Web-Based%20Survey.pdf)
- [_Battineni, G., Chintalapudi, N., & Amenta, F. (2020, June). AI Chatbot Design during an Epidemic Like the Novel Coronavirus. In Healthcare (Vol. 8, No. 2, p. 154). Multidisciplinary Digital Publishing Institute._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/AI%20Chatbot%20Design%20during%20an%20Epidemic%20like%20the%20Novel%20Coronavirus.pdf)
- [_Nadarzynski, T., Miles, O., Cowie, A., & Ridge, D. (2019). Acceptability of artificial intelligence (AI)-led chatbot services in healthcare: A mixed-methods study. Digital health, 5, 2055207619871808._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/Acceptability%20of%20artificial%20intelligence%20(AI)-led%20chatbot%20services%20in%20healthcare.pdf)
