import sklearn
import pandas as pd
import seaborn as sns

df = sns.load_dataset("penguins")
df = df.dropna()

df['target'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
categorical = ['island', 'sex']
numerical = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=1)

y_train = df_train.target.values
y_val = df_val.target.values

train_dict = df_train[categorical + numerical].to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train_cat = dv.transform(train_dict)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(df_train[numerical])

X_train_num = sc.transform(df_train[numerical])

import numpy as np
X_train = np.hstack([X_train_cat, X_train_num])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)

from sklearn.svm import SVC
svm = SVC(kernel='linear',C=1.0, random_state=1, probability=True)
svm.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
dt.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train, y_train)


import pickle

with open('./models/lr.pck', 'wb') as f:
    pickle.dump((dv, sc, lr), f)

with open('./models/svm.pck', 'wb') as f:
    pickle.dump((dv, sc, svm), f)

with open('./models/dt.pck', 'wb') as f:
    pickle.dump((dv, sc, dt), f)

with open('./models/knn.pck', 'wb') as f:
    pickle.dump((dv, sc, knn), f)
