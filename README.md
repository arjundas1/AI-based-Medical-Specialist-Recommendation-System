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
1. Python Language Libraries:
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - Scikitlearn
2. HTML/CSS

## Implementation

### Dataset
The dataset that will be used for this project is a publicly available dataset that was found on Kaggle, a Machine Learning and Data Science Community platform. It is created by Pranay Patil and Pratik Rathod. The dataset can be found in the [Dataset](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/tree/main/Dataset) folder of this respository. It is a very new dataset as it was uploaded in 2019 and updated in 2020. The usability of this dataset is rated very high by Kaggle (9.7/10). There are four csv files in the Dataset folder:

- [Symptom.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom.csv): This is the most important csv file, where the symptoms and their corresponding disease has been mentioned. The dataset consists of 17 columns of symptoms and 1 column for Disease name. The symptom columns (named as Symptom_1 to Symptom_17) has a lot of null values in their cells, indicating to the fact that a disease can have less than 17 symptoms.
- [Symptom Description.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Description.csv): This csv file contains the description of every unique disease that has been mentioned in the Symptom.csv file. Although the name of the file is Symptom description, it does not have any description of any of the symptom.
- [Symptom Severity.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Severity.csv): This csv file contains the level of severity of the symptoms that are present in Symptom.csv. The highest (indicating to most critical severity) symptom has been given 7 and the lease (indicating to least critical severity) has been given 1.
- [Symptom Precaution.csv](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Dataset/Symptom%20Precaution.csv): This csv file contains 4 columns of simple precautions that can be taken to avoid contracting the list of unique diseases present in Symptom.csv.

### Reading the data
We read the dataset using Python's pandas library
```python
import pandas as pd
df1 = pd.read_csv('Symptom.csv')
df2 = pd.read_csv('Symptom Description.csv')
df3 = pd.read_csv('Symptom Precaution.csv')
df4 = pd.read_csv('Symptom Severity.csv')
df1.head()
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
%matplotlib inline
sns.set()
```
 2. Visualising the count of various severity levels of symptoms using a histogram
```python
vis1 = plt.hist(df4["weight"], color=sns.color_palette()[8])
plt.xlabel("Count")
plt.ylabel("Severity")
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Severity-count%20histogram.png" width="350" height="250">
  </a>
</p>
From this figure, we infer that most of the symptoms mentioned in this datset has severity levels 4 and 5

3. In order to understand the various symptoms that are present in the columns, we can make pie charts of every to see the variety of symptoms present in each column. For easy understanding, we have included a ring plot of 'Symptom_11' column
```python
symptom = df1["Symptom_12"].value_counts()
vis2 = plt.pie(symptom, labels=symptom.index, startangle=100, 
               counterclock=False, wedgeprops={'width': 0.4})
plt.title("Symptom 12")
```
<p align="left">
  <a>
    <img src="https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/Files/Ring%20plot%20Symptom%2012.png" width="360" height="260">
  </a>
</p>

### Data Preprocessing
1. From the above head of df1 dataframe, we observe that a lot of Null values are present in the dataset. Therefore we count the total number of NaN values in the dataset.
```python
df1.isna().count()
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

2. Due to the large number of null values from Sypmtom_6 onwards, we will not be using the columns from Symptom_6 onwards in our Machine Learning model. We also combine the other csv files into a single dataframe.
```python
combined_df = pd.merge(df1, df2, on = 'Disease')
combined_df = pd.merge(combined_df , df3, on = 'Disease')
x = combined_df[['Symptom_1', 'Symptom_2', 'Symptom_3','Symptom_4','Symptom_5']]
```

3. For the machine to understand the column contents, we need to perform label encoding on these columns. We use scikitlearn's Label Encoder function for that. 
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in x.columns:
    x[i] = le.fit_transform(x[i].astype(str))
y = combined_df['Disease']
```

### ML Model
In order to find out the suitable machine learning algorithm, we have implemented several algorithms to determine the one yielding highest prediction accuracy. The conditions for train-test split has also been varied according to our understanding of effiency yield. However, every model uses 20% data as testing set.
```python
from sklearn.model_selection import train_test_split
```
#### Logistic Regression
Although the name has regression in it, Logistic regression is a classification algorithm. We identified the most efficient solver for our dataset to be newton-cg. Since covergence passes at 3000 iterations, the algorithm is slow with not a high prediction accuracy score.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=10)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="newton-cg",max_iter=3000)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print(round(lracc*100, 3), "%", sep="")
```
```
87.719%
```
We do not wish to choose Logistic Regression as the preferred ML Algortihm for our model

#### Support Vector Machine
SVM gives a very high prediction accuracy when implemented. However, this algorithm is very slow. This is possibly because of the eager learning construct of this algorithm, or that finding a suitable hyperplane is time consuming.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
from sklearn import svm, metrics
svmmodel = svm.SVC(kernel="linear", C=2)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print(round(svmacc*100, 3), "%", sep="")
```
```
97.697%
```
We do not wish to implement a time consuming algorithm like SVM, therefore we try other algorithms

#### K-Nearest Neighbors
KNN is a lazy learning algorithm, that yield very high prediction accuracy on the testing dataset. This is a very suitable algorithm for our purpose.
```python
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(trainX, trainY)
y_pred = classifier.predict(testX)
knnacc = classifier.score(testX, testY)
print(round(knnacc*100, 3),"%", sep="")
```
```
99.781%
```
Although this algorithm yields a very high prediction accuracy, we would try one last algorithm to see if we get higher prediction accuracy

#### Random Forest
Random Forest is based on Decision trees concepts where a number of Decision Trees are implemented and the best one is used. We have set number of estimators at 100.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print(round(rf.score(x_test,y_test) * 100, 3),"%",sep="")
```
```
100.0%
```
Since we are getting 100% prediction accuracy without any overfitting, we decide to use Random Forest further in this project.

## References
- [_Baclic, O., Tunis, M., Young, K., Doan, C., Swerdfeger, H., & Schonfeld, J. (2020). Artificial intelligence in public health: Challenges and opportunities for public health made possible by advances in natural language processing. Canada Communicable Disease Report, 46(6), 161._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/Challenges%20and%20opportunities%20for%20public%20health.pdf)
- [_Harsh, M., Suhas, D., Manthan, T., Anas, D. (2021). AI Based Healthcare Chatbot System by Using Natural Language. International Journal of Scientific Research and Engineering Development, Volume 4 Issue 2._](https://github.com/arjundas1/AI-based-Medical-Specialist-Recommendation-System/blob/main/References/AI%20Based%20Healthcare%20Chatbot%20System%20by%20Using%20Natural%20Language.pdf)
- _Khanna, A., Pandey, B., Vashishta, K., Kalia, K., Pradeepkumar, B., & Das, T. (2015). A study of today’s ai through chatbots and rediscovery of machine intelligence. International Journal of u-and e-Service, Science and Technology, 8(7), 277-284._
- _Palanica, A., Flaschner, P., Thommandram, A., Li, M., & Fossat, Y. (2019). Physicians’ perceptions of chatbots in health care: cross-sectional web-based survey. Journal of medical Internet research, 21(4), e12887._
- _Battineni, G., Chintalapudi, N., & Amenta, F. (2020, June). AI Chatbot Design during an Epidemic Like the Novel Coronavirus. In Healthcare (Vol. 8, No. 2, p. 154). Multidisciplinary Digital Publishing Institute._
- _Nadarzynski, T., Miles, O., Cowie, A., & Ridge, D. (2019). Acceptability of artificial intelligence (AI)-led chatbot services in healthcare: A mixed-methods study. Digital health, 5, 2055207619871808._
