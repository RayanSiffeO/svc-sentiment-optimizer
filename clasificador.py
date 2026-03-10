import joblib
import os
import subprocess

# 1. Definimos las rutas
MODEL_PATH = 'models/modelo_sentimientos.pkl'
VECTORIZER_PATH = 'models/vectorizador.pkl'

def verificar_y_entrenar():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Modelo no encontrado. Iniciando entrenamiento automático...")
        subprocess.run(["python", "entrenamiento.py"], check=True)
        print("Entrenamiento finalizado. Cargando modelo...")

def clasificar_frase(texto):

    verificar_y_entrenar()
    

    modelo = joblib.load(MODEL_PATH)
    vectorizador = joblib.load(VECTORIZER_PATH)
    
    vector_texto = vectorizador.transform([texto])
    prediccion = modelo.predict(vector_texto)
    return prediccion[0]

if __name__ == "__main__":
    frase = "This movie was absolutely amazing and changed my life!"
    print(f"Sentimiento detectado: {clasificar_frase(frase)}")