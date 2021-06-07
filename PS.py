import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import imblearn
from imblearn.over_sampling import SMOTE, ADASYN,BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


filename = 'clubhouse-scores4.csv'
raw_data = open(filename, 'rt')
reader1 = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
td1 = list(reader1)

print()
 # select columns 1 through end
data1 = np.array(td1).astype('float')
X=data1[:,1:10]
y=data1[:,0]
y = np.array(y).astype(int)
print(y)
kf=KFold(n_splits=10, random_state=None, shuffle=True)

acc=[]
prec=[]
reca=[]
f=[]


for train_index, test_index in kf.split(X):
        
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(y_train)
        clf= XGBClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict( X_test)
        acc.append(accuracy_score(y_test,  y_pred))
        reca.append(recall_score(y_test,  y_pred))
        prec.append(precision_score(y_test,  y_pred))
        f.append(f1_score(y_test,  y_pred))
        
        
print("accuracy : "+ str(np.mean(acc)))
print("recall : "+ str(np.mean(reca)))
print("precision : "+ str(np.mean(prec)))
print("f1 score : " + str(np.mean(f)))


