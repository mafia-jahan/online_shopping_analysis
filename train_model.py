import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import permutation_importance

df = pd.read_csv('Online_Shopping_Data.csv')
n0 = len(df)
# cleaning
df = df[df['Online_Shopping_Freq'].isin(['Once in a month','Rarely or Never','Once in a week','Multiple times per week'])]
df = df[df['Gender'].isin(['Male','Female','Other'])]
df['Gender'] = df['Gender'].replace({'Other':'Other/Unspecified'})
top = df['Frequent_Products'].value_counts()
df['Frequent_Products'] = df['Frequent_Products'].where(df['Frequent_Products'].isin(top.index[:4]), 'Other')
df['Major_Drawback'] = df['Major_Drawback'].where(df['Major_Drawback'].str.startswith(('Yes','No,')), 'Other')
df['Desired_Improvements'] = df['Desired_Improvements'].apply(lambda s: s if s in ['Better product descriptions and images','Enhanced customer reviews and ratings system','More personalized recommendations','Improved search and filtering options'] else 'Multiple/Other')
df['Local_vs_Intl_Retailers'] = df['Local_vs_Intl_Retailers'].str.replace('lnternational','International')
n1 = len(df)
df['Regular_Shopper'] = (df['Online_Shopping_Freq']!='Rarely or Never').astype(int)
y = df['Regular_Shopper']
X = df.drop(columns=['Regular_Shopper','Online_Shopping_Freq','Online_Purchase_Proportion'])
cats = list(X.columns)
pre = ColumnTransformer([('oh', OneHotEncoder(handle_unknown='ignore'), cats)])
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
cv = StratifiedKFold(5,shuffle=True,random_state=42)
models = {
 'Baseline (majority)': DummyClassifier(strategy='most_frequent'),
 'Logistic Regression': LogisticRegression(max_iter=2000,C=0.3,class_weight='balanced'),
 'Random Forest': RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight='balanced',random_state=42),
 'Gradient Boosting': GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05,random_state=42),
}
res=[]; fitted={}
for name,m in models.items():
    p = Pipeline([('pre',pre),('m',m)])
    cvs = cross_val_score(p,Xtr,ytr,cv=cv,scoring='roc_auc')
    cva = cross_val_score(p,Xtr,ytr,cv=cv,scoring='accuracy')
    p.fit(Xtr,ytr); fitted[name]=p
    pr = p.predict(Xte)
    proba = p.predict_proba(Xte)[:,1]
    res.append(dict(Model=name,CV_AUC=cvs.mean(),CV_AUC_sd=cvs.std(),CV_Acc=cva.mean(),Test_Acc=accuracy_score(yte,pr),Test_F1=f1_score(yte,pr),Test_AUC=roc_auc_score(yte,proba)))
R = pd.DataFrame(res); print(R.round(3).to_string())
# tune RF
gs = GridSearchCV(Pipeline([('pre',pre),('m',RandomForestClassifier(class_weight='balanced',random_state=42))]),
   {'m__n_estimators':[200,400],'m__max_depth':[3,5,None],'m__min_samples_leaf':[2,4,8]},cv=cv,scoring='roc_auc')
gs.fit(Xtr,ytr); best=gs.best_estimator_
pr=best.predict(Xte); proba=best.predict_proba(Xte)[:,1]
tuned=dict(params=gs.best_params_,cv_auc=gs.best_score_,acc=accuracy_score(yte,pr),f1=f1_score(yte,pr),auc=roc_auc_score(yte,proba))
print(tuned); print(classification_report(yte,pr,target_names=['Rarely/Never','Regular']))
cr = classification_report(yte,pr,target_names=['Rarely/Never','Regular'],output_dict=True)
# figures
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(5,3)); df['Online_Shopping_Freq'].value_counts().plot.barh(ax=ax,color='#2f5d8a'); ax.invert_yaxis(); ax.set_xlabel('Respondents'); ax.set_ylabel(''); plt.tight_layout(); plt.savefig('f_target.png',dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(6,3.2)); x=np.arange(len(R)); ax.bar(x-0.2,R.CV_AUC,0.4,label='CV ROC-AUC',color='#2f5d8a'); ax.bar(x+0.2,R.Test_AUC,0.4,label='Test ROC-AUC',color='#e08a2c'); ax.axhline(0.5,color='grey',ls='--',lw=0.8); ax.set_xticks(x); ax.set_xticklabels([m.replace(' ','\n') for m in R.Model],fontsize=8); ax.set_ylim(0.3,0.9); ax.legend(frameon=False,fontsize=8); plt.tight_layout(); plt.savefig('f_models.png',dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(3.6,3.2)); ConfusionMatrixDisplay(confusion_matrix(yte,pr),display_labels=['Rarely/Never','Regular']).plot(ax=ax,cmap='Blues',colorbar=False); plt.tight_layout(); plt.savefig('f_cm.png',dpi=160); plt.close()
pi = permutation_importance(best,Xte,yte,scoring='roc_auc',n_repeats=50,random_state=42)
imp = pd.Series(pi.importances_mean,index=cats).sort_values(); print(imp.round(3).to_string())
fig,ax=plt.subplots(figsize=(6,4)); imp.plot.barh(ax=ax,color='#2f5d8a'); ax.axvline(0,color='k',lw=0.6); ax.set_xlabel('Drop in test ROC-AUC when shuffled'); plt.tight_layout(); plt.savefig('f_imp.png',dpi=160); plt.close()
# descriptive: regular share by category for top features
desc={}
for c in imp.index[-3:][::-1]:
    desc[c]=(df.groupby(c)['Regular_Shopper'].agg(['mean','count']).round(2)).to_dict('index')
print(json.dumps(desc,indent=1))
json.dump(dict(n0=n0,n1=n1,pos=int(y.sum()),neg=int((1-y).sum()),R=R.round(3).to_dict('records'),tuned=tuned,cr=cr,imp=imp.round(4).to_dict(),desc=desc,ntr=len(Xtr),nte=len(Xte)),open('res.json','w'),default=str)
import joblib; joblib.dump(best,'shopping_model.joblib')
joblib.dump(fitted['Logistic Regression'],'shopping_model.joblib')
lr=fitted['Logistic Regression']; prl=lr.predict(Xte)
print(classification_report(yte,prl,target_names=['Rarely/Never','Regular']))
json.dump(classification_report(yte,prl,output_dict=True),open('cr_lr.json','w'))
print(confusion_matrix(yte,prl))
fig,ax=plt.subplots(figsize=(3.6,3.2)); ConfusionMatrixDisplay(confusion_matrix(yte,prl),display_labels=['Rarely/Never','Regular']).plot(ax=ax,cmap='Blues',colorbar=False); plt.tight_layout(); plt.savefig('f_cm.png',dpi=160); plt.close()
