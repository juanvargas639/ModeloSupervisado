import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from difflib import get_close_matches
import pandas as pd

# Función para cargar el dataset desde un archivo CSV
def cargar_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append(row)
    return dataset

# Función para encontrar la estación más cercana dado un nombre
def encontrar_estacion_cercana(nombre_estacion):
    estaciones = [viaje["origen"] for viaje in dataset] + [viaje["destino"] for viaje in dataset]
    coincidencias = get_close_matches(nombre_estacion, estaciones, n=1, cutoff=0.6)
    if coincidencias:
        return coincidencias[0]
    else:
        return None

# Cargar el dataset desde el archivo CSV
dataset = cargar_dataset('dataSet.csv')

# Dividir el dataset en conjunto de entrenamiento y prueba
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Extraer características y etiquetas del conjunto de entrenamiento
X_train = [[int(viaje["hora"])] for viaje in train_data]
y_train = [viaje["satisfaccion"] for viaje in train_data]

# Convertir X_train a un DataFrame de pandas
X_train_df = pd.DataFrame(X_train, columns=["hora"])

# Convertir la característica de "dia_semana" en variables dummy
X_train_encoded = pd.get_dummies(X_train_df)

# Hacer lo mismo para X_test
X_test = [[int(viaje["hora"])] for viaje in test_data]
X_test_df = pd.DataFrame(X_test, columns=["hora"])
X_test_encoded = pd.get_dummies(X_test_df)

# Entrenar modelo de aprendizaje supervisado (por ejemplo, un clasificador de bosque aleatorio)
model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)

# Evaluar el modelo
y_test = [viaje["satisfaccion"] for viaje in test_data]
y_pred = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Ahora, podemos usar este modelo para recomendar la mejor hora y preferencia de transporte para un nuevo viaje
def recomendar_ruta(origen, destino, dia):
    # Convertir nombres de estaciones a las más cercanas en el dataset
    origen_cercano = encontrar_estacion_cercana(origen)
    destino_cercano = encontrar_estacion_cercana(destino)
    if not origen_cercano or not destino_cercano:
        return "No se encontraron estaciones cercanas."
    
    # Recuperar los registros de viajes entre las estaciones de origen y destino
    registros = [viaje for viaje in dataset if viaje["origen"] == origen_cercano and viaje["destino"] == destino_cercano]

    # Calcular la hora promedio de los viajes y la satisfacción promedio
    horas = [int(viaje["hora"]) for viaje in registros]
    hora_promedio = sum(horas) / len(horas)
    satisfacciones = [viaje["satisfaccion"] for viaje in registros]
    satisfaccion_promedio = sum([1 if s == "Alta" else 0.5 if s == "Media" else 0 for s in satisfacciones]) / len(satisfacciones)

    # Determinar la preferencia de transporte en base a la satisfacción promedio
    preferencia = "Transitada" if satisfaccion_promedio >= 0.6 else "Económica"

    return hora_promedio, preferencia

# Ejemplo de uso
origen = input("Ingrese la estación de origen: ")
destino = input("Ingrese la estación de destino: ")
dia = input("Ingrese el día de la semana: ")
hora_recomendada, preferencia_recomendada = recomendar_ruta(origen, destino, dia)
print(f"Se recomienda tomar el transporte a las {hora_recomendada:.2f} horas con preferencia {preferencia_recomendada}.")
