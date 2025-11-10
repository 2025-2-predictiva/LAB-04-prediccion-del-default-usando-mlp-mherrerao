# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del psago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os 
import pandas as pd
import gzip
import json
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier


# Cargar la data
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False)
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False)


# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
train_data.rename(columns={'default payment next month':'default'}, inplace=True)
test_data.rename(columns={'default payment next month':'default'}, inplace=True)

# - Remueva la columna "ID".
train_data.drop(columns='ID', inplace=True)
test_data.drop(columns='ID', inplace=True)

# - Elimine los registros con informacion no disponible.
train_data = train_data[(train_data['EDUCATION'] != 0)]
train_data = train_data[(train_data['MARRIAGE'] != 0)]
test_data= test_data[(test_data['EDUCATION'] != 0)]
test_data= test_data[(test_data['MARRIAGE'] != 0)] 

# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
train_data['EDUCATION'] = train_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
test_data['EDUCATION'] = test_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)


# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = train_data.drop(columns='default')
y_train = train_data["default"]

x_test = test_data.drop(columns='default')
y_test = test_data["default"]


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.

def make_pipeline():

    categorical_features=["SEX","EDUCATION","MARRIAGE"]
    numerical_features = [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('scaler',StandardScaler(with_mean=True, with_std=True),numerical_features),
        ],
    )

    pipeline=Pipeline(
        [
            ("preprocessor",preprocessor),
            ('feature_selection',SelectKBest(score_func=f_classif)),
            ('pca',PCA()),
            ('classifier',MLPClassifier(max_iter=1000,random_state=17))
        ]
    )
    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

def make_grid_search(pipeline, x_train, y_train):

    param_grid = {
        'pca__n_components': [None], # le dice al objeto PCA() dentro del pipeline qué número de componentes conservar.
        'feature_selection__k':[20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60)], # Es una tupla que especifica la cantidad de neuronas en cada capa oculta.
        'classifier__alpha': [0.26], # coeficiente de regularización L2 (penalización) en MLPClassifier.
        "classifier__learning_rate_init": [0.001], # tasa de aprendizaje inicial para el optimizador de la red (cómo de grandes son los pasos en la actualización de pesos).
    }

    model = GridSearchCV(
            pipeline,
            param_grid,
            scoring='balanced_accuracy',  
            cv=10,                                        
            n_jobs=-1,                                  
            verbose=1,
            refit=True                              
    )

    model.fit(x_train, y_train)

    return model


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.}

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)
    model_file = os.path.join(models_path, "model.pkl.gz")

    with gzip.open(model_file, "wb") as file:
        pickle.dump(estimator, file)


# Paso 6 y 7

def save_metrics(model, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)


    # Métricas
    resultados = [
        # Métricas train
        {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)  
        },
         # Métricas test
        {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        },
        # Matriz confusión train
      {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0][0]), "predicted_1": int(cm_train[0][1])},
        "true_1": {"predicted_0": int(cm_train[1][0]), "predicted_1": int(cm_train[1][1])}
    },
     # Matriz confusión train
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0][0]), "predicted_1": int(cm_test[0][1])},
        "true_1": {"predicted_0": int(cm_test[1][0]), "predicted_1": int(cm_test[1][1])}
    }
    ]

    with open("files/output/metrics.json", "w") as f:
        for item in resultados:
            json.dump(item, f)
            f.write("\n")


    return resultados

def main():
        pipeline = make_pipeline()
        model = make_grid_search(pipeline, x_train, y_train)
        save_estimator(model)
        save_metrics(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()