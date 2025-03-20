import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger les données
application_train = pd.read_csv('data/application_train.csv')

# Exploration des données (afficher les premières lignes et la description)
print(application_train.info())
print(application_train.describe())

# Nettoyage des données : gestion des valeurs manquantes
for col in application_train.select_dtypes(include=np.number):
    application_train[col].fillna(application_train[col].mean())

for col in application_train.select_dtypes(include=object):
    application_train[col].fillna('Missing')

# Encodage des variables catégorielles
le = LabelEncoder()
for col in application_train.select_dtypes(include=object):
    application_train[col] = le.fit_transform(application_train[col])

# Séparer les caractéristiques (X) et la cible (y)
X = application_train.drop('TARGET', axis=1)
y = application_train['TARGET']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données numériques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sauvegarder le scaler et les données traitées
pd.to_pickle(scaler, 'scaler.pkl')
pd.to_pickle(X_train, 'X_train.pkl')
pd.to_pickle(X_test, 'X_test.pkl')
pd.to_pickle(y_train, 'y_train.pkl')
pd.to_pickle(y_test, 'y_test.pkl')
