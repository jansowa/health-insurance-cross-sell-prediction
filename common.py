import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import time

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance, plot_tree, to_graphviz
import xgboost
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
import warnings
import math
import seaborn as sns
import optuna
import graphviz
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# variables from data_analysis.ipynb

le = LabelEncoder()
X_le = X.copy()
X_le["Gender"] = le.fit_transform(X["Gender"])
X_le["Vehicle_Age"] = le.fit_transform(X["Vehicle_Age"])
X_le["Vehicle_Damage"] = le.fit_transform(X["Vehicle_Damage"])
df_le = pd.concat([X_le, y], axis=1)

# common ML methods
def print_confusion_matrix(y_true, classifier, X_test):
    y_pred = classifier.predict(X_test)
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                   s=confmat[i,j],
                   va='center', ha='center')
    plt.xlabel('Predicted class')
    plt.ylabel('Real class')
    plt.show()

def cross_val_summary(classifier, X, y, cv=10, scoring='roc_auc', message=None):
    start_time = time.time()
    score = cross_val_score(classifier, X, y, cv=cv, scoring=scoring).mean()
    print("Cross validation time: %s seconds" % (time.time() - start_time))
    if message is None:
      print("Mean score:", score)
    else:
      print(message + str(score))
