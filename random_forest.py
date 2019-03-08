import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def format_data():
    file = 'data/data_features.csv'
    df = pd.read_csv(file, sep=";", header=0, encoding="ISO-8859-1", low_memory=False)

    df['iyear'] = df['iyear'].astype(int)
    df = df[df['iyear'] >= 2000]
    df.drop(['Unnamed: 0', 'iyear'], axis=1, inplace=True)
    df.fillna(-99, inplace=True)

    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)

    print(list(df.columns))
    print(df.head(5))

    df.to_csv('data/data_ml.csv', sep=';', encoding='utf-8')


# format_data()

def get_train_test():
    file = 'data/data_ml.csv'
    df = pd.read_csv(file, sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
    df_X = df.drop('gname', axis=1)
    df_y = df['gname']

    return train_test_split(df_X, df_y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = get_train_test()
# X_train = X_train.values
# y_train = y_train.values

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=10)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X_train, y_train)

