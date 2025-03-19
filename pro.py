#Fake news detection
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")
df_fake["class"]=0
df_true["class"]=1
df_fake_manual_testing=df_fake.tail(10)
for i in range(23480,23740,-1):
    df_fake.drop([i],axis=0,inplace=True)
df_true_manual_testing=df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i],axis=0,inplace=True)
df_manual_testing=pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv("manual_testing.csv")
df_merge=pd.concat([df_fake,df_true],axis=0)
#print(df_merge.head(10))
df=df_merge.drop(["title","subject","date"],axis=1)
#print(df.head(10))
df=df.sample(frac=1)
#print(df.head())
#print(df.isnull().sum())
def word_drop(text): #remove all unnecessary character and links
    text=text.lower()
    text=re.sub(r'\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub(r'https?://\S+|www\.\S+', '',text)
    text=re.sub('<.*?>+', '',text)
    text=re.sub('[%s]'%re.escape(string.punctuation), '',text)
    text=re.sub(r'\w*\d\w*', '',text)
    return text
df["text"]=df["text"].apply(word_drop)
#print(df.head(10)) removed all unnecessary characters
x=df["text"]
y=df["class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=25)
#vectorising variable 
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)
#logistic 
LR=LogisticRegression()
LR.fit(xv_train,y_train)
#print(LR.score(xv_test,y_test))#to know accuracy
pred_LR=LR.predict(xv_test)
#print(classification_report(y_test,pred_LR))
#decision tree classification
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
#print(DT.score(xv_test,y_test))
pred_DT=DT.predict(xv_test)
#print(classification_report(y_test,pred_DT)) accuracy is 100%
GBC=GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)
#print(GBC.score(xv_test,y_test)) #accuracy is 100%
pred_GBC=GBC.predict(xv_test)
#print(classification_report(y_test,pred_GBC)) accuracy is 96%
RFC=RandomForestClassifier()
RFC.fit(xv_train,y_train)
#print(RFC.score(xv_test,y_test)) accuracy is 100
pred_RFC=RFC.predict(xv_test)
print(classification_report(y_test,pred_RFC))
#manual testing
def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not a Fake News"
def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test['text']=new_def_test["text"].apply(word_drop)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    pred_GBC=GBC.predict(new_xv_test)
    pred_RFC=RFC.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction : {}".format(output_lable(pred_LR),output_lable(pred_DT),output_lable(pred_GBC),output_lable(pred_RFC)))
news=str(input("Enter news:"))
manual_testing(news)