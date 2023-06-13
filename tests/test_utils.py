import numpy as np
import pandas as pd
from sklearn.svm import SVC

import pickle
import os
import random

def get_predictions(df_left, df_right):
    np.random.seed(42)
    random.seed(42)

    classifier_left = SVC(gamma='auto')
    classifier_right = SVC(gamma='auto')

    classifier_left.fit(df_left[['avar', 'bvar', 'cvar']], df_left['category'])
    classifier_right.fit(df_right[['bvar', 'cvar', 'dvar']], df_right['category'])

    predicted_labels_t1 = classifier_left.predict(df_left[['avar', 'bvar', 'cvar']])
    predicted_labels_t2 = classifier_right.predict(df_right[['bvar', 'cvar', 'dvar']])

    return predicted_labels_t1, predicted_labels_t2


def setup_synthetic_text():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_left = pd.read_csv(f'{dir_path}/test_data/df_left.csv')
    df_right = pd.read_csv(f'{dir_path}/test_data/df_right.csv')

    features_t1 = ['avar', 'bvar', 'cvar']
    features_t2 = ['bvar', 'cvar', 'dvar']

    # Get model and predictions
    predicted_labels_t1, predicted_labels_t2 = get_predictions(df_left, df_right)

    for col in features_t1:
        df_left[col] = [col if x == 1 else '' for x in df_left[col]]
    for col in features_t2:
        df_right[col] = [col if x == 1 else '' for x in df_right[col]]

    df_left['corpus'] = df_left['avar'] + ' ' + df_left['bvar'] + ' ' + df_left['cvar']
    df_right['corpus'] = df_right['bvar'] + ' ' + df_right['cvar'] + ' ' + df_right['dvar']

    X_t1, X_t2 = df_left['corpus'], df_right['corpus']
    return (X_t1, predicted_labels_t1,
            X_t2, predicted_labels_t2)


def setup_synthetic_tabular():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_left = pd.read_csv(f'{dir_path}/test_data/df_left.csv')
    df_right = pd.read_csv(f'{dir_path}/test_data/df_right.csv')
    features_t1 = ['avar', 'bvar', 'cvar']
    features_t2 = ['bvar', 'cvar', 'dvar']

    # Get model and predictions
    predicted_labels_t1, predicted_labels_t2 = get_predictions(df_left, df_right)

    X_t1, X_t2 = df_left[features_t1], df_right[features_t2]
    return (X_t1, predicted_labels_t1,
            X_t2, predicted_labels_t2)
