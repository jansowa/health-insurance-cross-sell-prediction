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
from sklearn.model_selection import cross_val_score, train_test_split
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
from numpy.typing import *

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

le = LabelEncoder()
X_le = X.copy()
X_le["Gender"] = le.fit_transform(X["Gender"])
X_le["Vehicle_Age"] = le.fit_transform(X["Vehicle_Age"])
X_le["Vehicle_Damage"] = le.fit_transform(X["Vehicle_Damage"])
df_le = pd.concat([X_le, y], axis=1)


def cross_val_summary(classifier, X: ArrayLike, y: ArrayLike, cv: int = 10, scoring='roc_auc',
                      message: str = None) -> float:
    start_time = time.time()
    score = cross_val_score(classifier, X, y, cv=cv, scoring=scoring).mean()
    print("Cross validation time: %s seconds" % (time.time() - start_time))
    if message is None:
        print("Mean score:", score)
    else:
        print(message + str(score))
    return score


def underscore_to_newline(text: str) -> str:
    return text.replace("_", "\n")


def compare_preprocessing_te(model, X, y, te_columns) -> float:
    encoder = TargetEncoder(cols=te_columns)

    scaler = StandardScaler()
    transformer = PowerTransformer()
    scaler_without_std = StandardScaler(with_std=False)
    quant_trans_uniform = QuantileTransformer(output_distribution='uniform')
    quant_trans_normal = QuantileTransformer(output_distribution='normal')

    model_te = make_pipeline(encoder, model)
    model_scaler = make_pipeline(encoder, scaler, model)
    model_transformer = make_pipeline(encoder, transformer, model)
    model_scaler_transformer = make_pipeline(encoder, scaler_without_std, transformer, model)
    model_quant_trans_uniform = make_pipeline(encoder, quant_trans_uniform, model)
    model_quant_trans_normal = make_pipeline(encoder, quant_trans_normal, model)

    highest_score = cross_val_summary(model_te, X, y, message="Mean ROC_AUC score without scaler: ")
    highest_score = max(highest_score,
                        cross_val_summary(model_scaler, X, y, message="Mean ROC_AUC score with scaler: "))
    highest_score = max(highest_score,
                        cross_val_summary(model_transformer, X, y, message="Mean ROC_AUC score with transformer: "))
    highest_score = max(highest_score, cross_val_summary(model_scaler_transformer, X, y,
                                                         message="Mean ROC_AUC score with scaler and transformer: "))
    highest_score = max(highest_score, cross_val_summary(model_quant_trans_uniform, X, y,
                                                         message="Mean ROC_AUC score with QuantileTransformer (uniform distribution): "))
    highest_score = max(highest_score, cross_val_summary(model_quant_trans_normal, X, y,
                                                         message="Mean ROC_AUC score with QuantileTransformer (normal distribution): "))
    return highest_score


def save_print_result(base_score: float, columns_to_remove, full_cycle_iter: int, path: str, message: str) -> None:
    pd.Series(columns_to_remove).to_csv(path + str(full_cycle_iter) + '.csv')
    print()
    print("Score after " + str(full_cycle_iter) + " cycle: " + str(base_score))
    print(str(len(columns_to_remove)) + message)


def forward_selection(classifier, X, y, fixed_columns, columns_to_select, path: str) -> None:
    encoder = TargetEncoder(cols=['Policy_Sales_Channel', 'Region_Code'])
    transformer = PowerTransformer()
    model = make_pipeline(encoder, transformer, classifier)

    full_cycle_iter = 0
    columns_to_add = []

    while True:
        full_cycle_iter += 1
        columns_to_add_len_before_cycle = len(columns_to_add)
        print("Starting full cycle number " + str(full_cycle_iter))
        selection_columns = [col for col in columns_to_select if col not in fixed_columns and col not in columns_to_add]
        print("Number of iters in this cycle: " + str(len(selection_columns)))
        base_score = cross_val_score(model, X[fixed_columns + columns_to_add], y, cv=10, scoring='roc_auc',
                                     n_jobs=-1).mean()
        print("Calculated base score: " + str(base_score))
        counter = 0
        print("Finished iterations: ", end='')
        for column in selection_columns:
            score_with_column = cross_val_score(model, X[fixed_columns + columns_to_add + [column]], y, cv=10,
                                                scoring='roc_auc', n_jobs=-1).mean()
            counter += 1
            print(str(counter) + ",", end='')
            if score_with_column >= base_score:
                base_score = score_with_column
                columns_to_add += [column]

        if columns_to_add_len_before_cycle == len(columns_to_add):
            print()
            print("No columns added in this cycle")
            break

        save_print_result(base_score, columns_to_add, full_cycle_iter, path, " columns was added")


def choose_columns_to_transform(classifier, X) -> None:
    encoder = TargetEncoder(cols=['Region_Code', 'Policy_Sales_Channel'])
    full_cycle_iter = 0
    columns_without_transformer = []
    while True:
        all_columns = [x for x in X.columns.values if x not in columns_without_transformer]
        full_cycle_iter += 1
        print("Starting full cycle number " + str(full_cycle_iter))
        print("All columns number: " + str(len(all_columns)))
        columns_wo_tran_len_before_cycle = len(columns_without_transformer)
        transformer = ColumnTransformer(transformers=[('pwrtrans', PowerTransformer(), all_columns)],
                                        remainder='passthrough')
        model = make_pipeline(encoder, transformer, classifier)
        base_score = cross_val_score(model, X, y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
        print("Base score: " + str(base_score))
        counter = 0
        print("Iterations finished: ", end='')
        for column in all_columns:
            columns_for_transformer = [x for x in all_columns if x not in columns_without_transformer + [column]]
            transformer = ColumnTransformer(transformers=[('pwrtrans', PowerTransformer(), columns_for_transformer)],
                                            remainder='passthrough')
            model = make_pipeline(encoder, transformer, classifier)

            score = cross_val_score(model, X, y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
            counter += 1
            print(str(counter), end='')

            if score >= base_score:
                base_score = score
                print("Score for transformer without " + column + " column: " + str(score))
                columns_without_transformer += [column]

        if columns_wo_tran_len_before_cycle == len(columns_without_transformer):
            print("No columns added in this cycle")
            break

        pd.Series(columns_without_transformer).to_csv(
            'results/columns_without_transformer' + str(full_cycle_iter) + '.csv')
        print("Score after " + str(full_cycle_iter) + " cycle: " + str(base_score))
        print(str(len(columns_without_transformer)) + " columns was removed")
        print()


def choose_columns_to_drop(classifier, X, columns_for_transformer) -> None:
    encoder_columns = ['Region_Code', 'Policy_Sales_Channel']
    full_cycle_iter = 0
    columns_to_drop = []
    while True:
        all_columns = [x for x in X.columns.values if x not in columns_to_drop]
        full_cycle_iter += 1
        print("Starting full cycle number " + str(full_cycle_iter))
        print("All columns number: " + str(len(all_columns)))
        columns_to_drop_len_before_cycle = len(columns_to_drop)
        inner_columns_for_transformer = [elem for elem in columns_for_transformer if elem not in columns_to_drop]
        transformer = ColumnTransformer(transformers=[('pwrtrans', PowerTransformer(), inner_columns_for_transformer)], remainder='passthrough')
        inner_encoder_columns = [elem for elem in encoder_columns if elem not in columns_to_drop]
        encoder = TargetEncoder(cols=inner_encoder_columns)
        model = make_pipeline(encoder, transformer, classifier)
        base_score = cross_val_score(model, X.drop(columns=columns_to_drop), y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
        print("Base score: " + str(base_score))
        counter = 0
        print("Iterations finished: ", end='')
        for column in all_columns:
            inner_columns_for_transformer = [elem for elem in columns_for_transformer if elem not in columns_to_drop + [column]]
            transformer = ColumnTransformer(transformers=[('pwrtrans', PowerTransformer(), inner_columns_for_transformer)], remainder='passthrough')
            inner_encoder_columns = [elem for elem in encoder_columns if elem not in columns_to_drop + [column]]
            encoder = TargetEncoder(cols=inner_encoder_columns)
            model = make_pipeline(encoder, transformer, classifier)

            score = cross_val_score(model, X.drop(columns=columns_to_drop + [column]), y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
            counter += 1
            print(str(counter), end='')

            if score >= base_score:
                base_score = score
                print("Score for model without " + column + " column: " + str(score))
                columns_to_drop += [column]

        if columns_to_drop_len_before_cycle == len(columns_to_drop):
            print("No columns added in this cycle")
            break

        pd.Series(columns_to_drop).to_csv(
            'results/columns_to_drop' + str(full_cycle_iter) + '.csv')
        print("Score after " + str(full_cycle_iter) + " cycle: " + str(base_score))
        print(str(len(columns_to_drop)) + " columns was removed")
        print()

