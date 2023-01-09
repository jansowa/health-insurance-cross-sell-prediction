import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import time
import matplotlib.patches as patches

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc #, plot_roc_curve
from scikitplot.metrics import plot_roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

num_features = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
cat_features = ['Driving_License', 'Gender', 'Previously_Insured', 'Vehicle_Damage', 'Vehicle_Age']

# variables from data_analysis.ipynb

le = LabelEncoder()
X_le = X.copy()
X_le["Gender"] = le.fit_transform(X["Gender"])
X_le["Vehicle_Age"] = le.fit_transform(X["Vehicle_Age"])
X_le["Vehicle_Damage"] = le.fit_transform(X["Vehicle_Damage"])
df_le = pd.concat([X_le, y], axis=1)

# variables from health_insurance.ipynb

X_le_train, X_le_test, y_train, y_test = train_test_split(
     X_le, y, test_size=0.25, random_state=1, stratify=y)


part = 30000
X_le_train_part = X_le_train.iloc[:part]
y_train_part = y_train.iloc[:part]
X_le_test_part = X_le_test.iloc[:part]
y_test_part = y_test.iloc[:part]
df_train = pd.concat([X_le_train_part, y_train_part], axis=1)

X_log_an = X_le.copy()
X_log_an["Log_Annual_Premium"] = np.log(X_le["Annual_Premium"])

X_log_an_train, X_log_an_test, y_train, y_test = train_test_split(
     X_log_an, y, test_size=0.25, random_state=1, stratify=y)


X_log_an_train_part = X_log_an_train.iloc[:part]
y_train_part = y_train.iloc[:part]
X_log_an_test_part = X_log_an_test.iloc[:part]
y_test_part = y_test.iloc[:part]

simple_log_reg = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, n_jobs=-1))
balanced_log_reg = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, n_jobs=-1, class_weight='balanced'))

simple_svc = make_pipeline(StandardScaler(), SVC(random_state=0, verbose=1))
balanced_svc = make_pipeline(StandardScaler(), SVC(verbose=1, random_state=0, class_weight='balanced'))
svc_pwrtran = make_pipeline(PowerTransformer(), SVC(random_state=0, verbose=1))
tuned_svc = make_pipeline(StandardScaler(), SVC(random_state=0, verbose=1, C=50))

simple_xgb = make_pipeline(StandardScaler(), XGBClassifier(n_jobs=-1, random_state=0, n_estimators=100, use_label_encoder=False))
balanced_xgb = make_pipeline(StandardScaler(), XGBClassifier(n_jobs=-1, random_state=0, n_estimators=100, eval_metric='auc', scale_pos_weight = 7.159, use_label_encoder=False))
xgb_pwrtran = make_pipeline(PowerTransformer(), XGBClassifier(n_jobs=-1, random_state=0, n_estimators=100, use_label_encoder=False))
tuned_xgb = make_pipeline(PowerTransformer(), XGBClassifier(random_state=0, learning_rate= 0.1, max_depth= 2, n_estimators= 200, subsample= 0.75, use_label_encoder=False))

simple_rfc = make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=100))
balanced_rfc = make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=100, class_weight='balanced'))

simple_knc = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1))

simple_adaboost = make_pipeline(StandardScaler(), AdaBoostClassifier())

# common ML methods
def stratified_sample_df(df, col, frac, random_state=None):
    return df.groupby(col).apply(lambda x: x.sample(frac=frac, random_state=random_state))

def get_XY(df):
    return df.iloc[:,0:-1], df.iloc[:,-1]

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

def print_model_summary(y_true, classifier, X_test):
    probabilities = None
    try:
        probabilities = classifier.decision_function(X_test)
    except:
        probabilities = classifier.predict_proba(X_test)[:,1]
    print("ROC AUC score:", roc_auc_score(y_true, probabilities))
    plot_roc_curve(classifier, X_test, y_true)  
    print_confusion_matrix(y_true, classifier, X_test)

def fit_with_time(classifier, X, y):
    start_time = time.time()
    classifier.fit(X, y)
    print("Fitting time: %s seconds" % (time.time() - start_time))

def fit_and_summary(classifier, X_train, y_train, X_test, y_test):
    fit_with_time(classifier, X_train, y_train)
    print_model_summary(y_test, classifier, X_test)

def cross_val_summary(classifier, X, y, cv=10, scoring='roc_auc', message=None):
    start_time = time.time()
    score = cross_val_score(classifier, X, y, cv=cv, scoring=scoring).mean()
    print("Cross validation time: %s seconds" % (time.time() - start_time))
    if message is None:
      print("Mean score:", score)
    else:
      print(message + str(score))

def gridsearch_fit_summary(gridsearch, X, y):
    start_time = time.time()
    try:
        gridsearch.fit(X, y)
    except:
        gridsearch.fit(X.values, y)
    print("Grid search fitting time: %s seconds" % (time.time() - start_time))
    print("best model's score:", gridsearch.best_score_)
    print("best model's params:", gridsearch.best_params_)

def cross_val_roc_curve(classifier, X, y, cv=10, figsize=[6,6]):
    kfold = StratifiedKFold(n_splits=cv,shuffle=False)
    fig1 = plt.figure(figsize=figsize)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    fitting_time = 0
    predicting_time = 0
    for train,test in kfold.split(X,y):
        start_time = time.time()
        try:
            classifier.fit(X.iloc[train],y.iloc[train])
        except:
            classifier.fit(X.iloc[train].values, y.iloc[train].values)
        fitting_time += (time.time() - start_time)
        prob = None
        start_time = time.time()
        try:
            prob = classifier.decision_function(X.iloc[test])
        except:
            prob = classifier.predict_proba(X.iloc[test])[:,1]
        predicting_time += (time.time() - start_time)
        fpr, tpr, t = roc_curve(y.iloc[test], prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
        i= i+1

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.4f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    print("Mean fitting time:", fitting_time/cv)
    print("Mean predicting time:", predicting_time/cv)
    plt.show()

def model_summary(classifier, X, y, transformers=None, samples_number=40000, cv=10, figsize=[6,6], data_transform = None, drop_columns = None, le_columns = None, ohe_columns = None, print_params = True, print_roc=True):
    X_part = X.iloc[:samples_number]
    if not drop_columns is None:
        X_part = X_part.drop(columns=drop_columns)
    y_part = y.iloc[:samples_number]
    if not data_transform is None: # TODO: split to data_transform_start and data_transform_end
        if type(data_transform) is list:
            for single_transform in data_transform:
                X_part = single_transform(X_part)
        else:
            X_part = data_transform(X_part)

    if not le_columns is None:
        le = LabelEncoder()
        if type(le_columns) is list:
            for column in le_columns:
                try:
                    X_part[column] = le.fit_transform(X_part[column])
                except:
                    print("error")
        else:
            try:
                X_part[le_columns] = le.fit_transform(X_part[le_columns])
            except:
                print("error")
    if not ohe_columns is None:
        try:
            dummies = pd.get_dummies(X_part[ohe_columns])
            X_part = pd.concat([X_part.drop(columns=ohe_columns), dummies], axis=1)
        except: #implement case for list ohe_columns
            print("error")
    real_clf = classifier
    if not transformers is None:
        if type(transformers) is list:
            transformers = tuple(transformers)
            pipe_elem = (*transformers, classifier)
            real_clf = make_pipeline(*pipe_elem)
        else:
            pipe_elem = (transformers, classifier)
            real_clf = make_pipeline(*pipe_elem)
    if print_params:
        params = ""
        params += str(real_clf)
        if samples_number != 40000:
            params += ", samples_number: " + str(samples_number)
        if cv != 10:
            params += ", " + str(cv) + "-fold CV"
        if not data_transform is None:
            params += ", data_transform: " + str(data_transform)
        if not drop_columns is None:
            params += ", dropped: " + str(drop_columns)
        if not le_columns is None:
            params += ", LabelEncoder: " + str(le_columns)
        if not ohe_columns is None:
            params += ", OneHotEncoder: " + str(ohe_columns)
        
        print(params)
    if print_roc:
        cross_val_roc_curve(real_clf, X_part, y_part, cv=cv, figsize=figsize)
    else:
        cross_val_summary(real_clf, X_part, y_part, cv=cv)

def model_summary_grid(classifier, X, y, param_grid, scoring='roc_auc', n_jobs = -1, transformers=None, samples_number=40000, cv=10, figsize=[6,6], data_transform = None, drop_columns = None, le_columns = None, ohe_columns = None, print_params = True, print_roc = True):
    X_part = X.iloc[:samples_number]
    if not drop_columns is None:
        X_part = X_part.drop(columns=drop_columns)
    y_part = y.iloc[:samples_number]
    if not data_transform is None:
        if type(data_transform) is list:
            for single_transform in data_transform:
                X_part = single_transform(X_part)
        else:
            X_part = data_transform(X_part)

    if not le_columns is None:
        le = LabelEncoder()
        if type(le_columns) is list:
            for column in le_columns:
                try:
                    X_part[column] = le.fit_transform(X_part[column])
                except:
                    print("error")
        else:
            try:
                X_part[le_columns] = le.fit_transform(X_part[le_columns])
            except:
                print("error")
    if not ohe_columns is None:
        try:
            dummies = pd.get_dummies(X_part[ohe_columns])
            X_part = pd.concat([X_part.drop(columns=ohe_columns), dummies], axis=1)
        except: #implement case for list ohe_columns
            print("error")
    real_clf = classifier
    if not transformers is None:
        if type(transformers) is list:
            transformers = tuple(transformers)
            pipe_elem = (*transformers, classifier)
            real_clf = make_pipeline(*pipe_elem)
        else:
            pipe_elem = (transformers, classifier)
            real_clf = make_pipeline(*pipe_elem)
    if print_params:
        params = ""
        params += str(real_clf)
        if samples_number != 40000:
            params += ", samples_number: " + str(samples_number)
        if cv != 10:
            params += ", " + str(cv) + "-fold CV"
        if not data_transform is None:
            params += ", data_transform: " + str(data_transform)
        if not drop_columns is None:
            params += ", dropped: " + str(drop_columns)
        if not le_columns is None:
            params += ", LabelEncoder: " + str(le_columns)
        if not ohe_columns is None:
            params += ", OneHotEncoder: " + str(ohe_columns)
        
        print(params)
    gs = GridSearchCV(estimator = real_clf,
                 param_grid=param_grid,
                 scoring=scoring,
                 cv=StratifiedKFold(n_splits=cv,shuffle=False),
                 n_jobs=n_jobs)

    gridsearch_fit_summary(gs, X_part, y_part)

    if print_roc:
        cross_val_roc_curve(gs.best_estimator_, X_part, y_part, cv=cv, figsize=figsize)

def compare_preprocessing(classifier, param_grid, param_grid_transform, X=X, y=y, samples_number=40000):
    model_summary_grid(classifier, X, y, param_grid, le_columns = ['Gender', 'Vehicle_Damage', 'Vehicle_Age'], samples_number=samples_number)
    model_summary_grid(classifier, X, y, param_grid_transform, transformers=StandardScaler(), le_columns = ['Gender', 'Vehicle_Damage', 'Vehicle_Age'], samples_number=samples_number)
    model_summary_grid(classifier, X, y, param_grid_transform, transformers=PowerTransformer(), le_columns = ['Gender', 'Vehicle_Damage', 'Vehicle_Age'], samples_number=samples_number)
    model_summary_grid(classifier, X, y, param_grid, le_columns = ['Gender', 'Vehicle_Damage'], ohe_columns='Vehicle_Age', samples_number=samples_number)
    model_summary_grid(classifier, X, y, param_grid_transform, transformers=StandardScaler(), le_columns = ['Gender', 'Vehicle_Damage'], ohe_columns='Vehicle_Age', samples_number=samples_number)
    model_summary_grid(classifier, X, y, param_grid_transform, transformers=PowerTransformer(), le_columns = ['Gender', 'Vehicle_Damage'], ohe_columns='Vehicle_Age', samples_number=samples_number)

def compare_drop_column(classifier, param_grid, X=X, y=y, le_columns=None, ohe_columns=None, transformers = None, samples_number=4000):
    for column in X.columns:
        model_summary_grid(classifier, X, y, param_grid, transformers=transformers, le_columns = le_columns, ohe_columns=ohe_columns, samples_number=samples_number, drop_columns=column)

def transform_columns(X, transformer, columns):
    X_copy = X.copy()
    for column in columns:
        try:
            X_copy[column] = transformer.fit_transform(X_copy[column].values.reshape(-1, 1))
        except:
            print("Column", column, "not found.")
    return X_copy

