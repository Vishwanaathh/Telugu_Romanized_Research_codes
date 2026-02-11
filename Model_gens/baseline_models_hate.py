import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

le=LabelEncoder()
tfidf=TfidfVectorizer(max_features=5000)

hate=pd.read_csv('../Datasets/cleaned_training_data_telugu-hate.csv')


hate['hate_encoded']=le.fit_transform(hate['Label'])
X=tfidf.fit_transform(hate['Comments'])
Y=hate['hate_encoded']

lr=LogisticRegression(max_iter=1000)
lr.fit(X,Y)
joblib.dump(lr,'Logregmodel.pkl')

rf=RandomForestClassifier(n_estimators=100,random_state=20)
rf.fit(X,Y)
joblib.dump(rf,'rfc.pkl')
