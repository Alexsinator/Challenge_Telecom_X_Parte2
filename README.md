El propósito del análisis realizado.

El propósito es anticiparse a la cancelación de los clientes, ayudando así a mantener una mayor rentabilidad en la empresa TelecomX


Estructura del proyecto.

- Lectura de los datos
  -   Análisis de los datos
  -   Revisión de consistencia
- Tratamiento
  - Codificación
  - Separacion de Train y Test
  - Escalado de valores
- Correlacion y Selección de variables
  - Analisis
  - SMOTE
- Modelos
  - DUMMY
  - Logistic Regression
  - KNeighbors Classifier
  - Random Forest Classifier
  - Decision Tree Classifier
  - Evaluación completa de los modelos
  - Comparativa de modelos
  - XGBosst
  - Suport Vector Machine
- Interpretacion y conclusiones
  - Informe

Descripción del proceso de preparación de los datos, incluyendo:

Se reviso la base de datos que ya se tenía tratada, sin embargo se busco tener la mayor fiabilidad al analizar toda la base.
Se busco que no hubiera inconsistencias en los datos.
Se analizó toda la información para poder realizar una tranformacion de los datos a categoricos y
así tener una base confiable con la cual se pudiera trabajar, es decir que se volvio númerica toda nuestra basa para poder trabajarlo con nuestros modelos.
Se realizó una trasformación de los datos para normalizar y poder codificar toda la base.

Separación de los datos en conjuntos de entrenamiento y prueba.


Instrucciones para ejecutar el cuaderno, incluyendo qué bibliotecas deben instalarse y cómo cargar los datos tratados.

Se instalaron las siguientes bibliotecas:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
