import joblib
import os
import subprocess

MODEL_PATH = 'models/modelo_sentimientos.pkl'
VECTORIZER_PATH = 'models/vectorizador.pkl'

modelo = None
vectorizador = None


def verificar_y_entrenar():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Modelo no encontrado. Iniciando entrenamiento ...")
        subprocess.run(["python", "entrenamiento.py"], check=True)
        print("Entrenamiento finalizado.")


def cargar_modelo():

    global modelo
    global vectorizador

    verificar_y_entrenar()

    modelo = joblib.load(MODEL_PATH)
    vectorizador = joblib.load(VECTORIZER_PATH)


def clasificar_frase(texto):

    vector_texto = vectorizador.transform([texto])
    prediccion = modelo.predict(vector_texto)

    return prediccion[0]


if __name__ == "__main__":

    cargar_modelo()

    reseñas = [
        "This movie was absolutely fantastic. The acting and story were incredible.",
        "I really enjoyed this film. The characters were interesting and the plot was engaging.",
        "One of the best movies I have seen in years.",
        "The cinematography was beautiful and the story was very emotional.",
        "Amazing performance by the lead actor.",
        "I loved every minute of this movie.",
        "The movie was entertaining and surprisingly touching.",
        "Great script and strong performances.",
        "A very enjoyable movie with a satisfying ending.",
        "This film was wonderful and inspiring.",

        "This movie was terrible. The plot made no sense.",
        "I regret watching this film. It was boring and too long.",
        "The acting was very bad and the story was predictable.",
        "One of the worst movies I have ever seen.",
        "The characters were poorly written.",
        "I couldn't finish the movie because it was so boring.",
        "The film had a good idea but the execution was awful.",
        "Everything about this movie felt cheap.",
        "The dialogue was cringe and badly written.",
        "A complete waste of time."
    ]

    for frase in reviews:
        print(f"Review: {frase}")
        print(f"Sentimiento detectado: {clasificar_frase(frase)}")
        print("-" * 50)
