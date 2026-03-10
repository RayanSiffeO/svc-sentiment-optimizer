import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Lectura y manipulacion de los datos
df_review = pd.read_csv("Excel/IMDB Dataset.csv")
#Input(x) - > Comentarios(review)
#Output(y) - > Sentimientos
redux = RandomUnderSampler()

df_positivo = df_review[df_review['sentiment']=='positive' ][:9000]
df_negativo = df_review[df_review['sentiment']=='negative' ][:1000]
matriz_valores = pd.concat([df_negativo,df_positivo])

df_reviewbal, df_reviewbal['sentiment'] = redux.fit_resample(matriz_valores[['review']], matriz_valores[['sentiment']])

print(df_reviewbal['sentiment'].value_counts())

#Campo de entrenamiento

entr, test = train_test_split(df_reviewbal, test_size=0.2, random_state=67)
entr_x , entr_y = entr['review'], entr['sentiment']
test_x, test_y = test['review'], test['sentiment']


#Palabras clave
text = ['Amo programar y escribir con python. Amo el codigo en python',
        'Odio programar y escribir en java. Odio el codigo en Java']
pc = pd.DataFrame({'review': ['reviewx','reviewy'], 'text':text})


#Por cantidad pruebas
cv = CountVectorizer()
cv_matrix = cv.fit_transform(pc['text'])
pc_dtm = pd.DataFrame(cv_matrix.toarray(), index = pc['review'].values , columns = cv.get_feature_names_out() )
#Por peso pruebas
tf = TfidfVectorizer(stop_words='english')
tf_matrix = tf.fit_transform(pc['text'])
tf_dtm = pd.DataFrame(tf_matrix.toarray(), index = pc['review'].values , columns = tf.get_feature_names_out() )
# Nota: Las palabras clave por peso son mas precisas que por cantidad (da importancia a palabras de relleno.)


#Transformacion de data string a integers
entr_vectorx = tf.fit_transform(entr_x)
test_vectorx = tf.transform(test_x)



# --- MACHINE LEARNING ---
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(entr_vectorx, entr_y)
#Metodo de arbol
from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(entr_vectorx, entr_y)
#Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(entr_vectorx, entr_y)
# Regresion logistica
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(entr_vectorx, entr_y)

#Evaluacion de modelos
from sklearn.metrics import accuracy_score

modelos = [("SVC", svc), ("Árbol", d_tree), ("Naive Bayes", mnb), ("Logística", lr)]

for nombre, modelo in modelos:
    pred = modelo.predict(test_vectorx)
    acc = accuracy_score(test_y, pred)
    print(f"Modelo {nombre} - Precisión: {acc:.4f}")
#Observaciones : El modelo de regresion es el que mas puntaje tiene, mientras que el de metodo arbol es el que menos tiene.
#F1 Score
from sklearn.metrics import classification_report

print(f"{'EVALUACIÓN DE MODELOS':^50}")
print("=" * 50)

for nombre, modelo in modelos:
    pred = modelo.predict(test_vectorx)
    
    print(f"\nResultados para: {nombre}")
    report = classification_report(test_y, pred, output_dict=True)
    
    f1_pos = report['positive']['f1-score']
    f1_neg = report['negative']['f1-score']
    
    print(f"F1-Score Positivo: {f1_pos:.4f}")
    print(f"F1-Score Negativo: {f1_neg:.4f}")
    print("-" * 30)
#Comprobaciones finales

from sklearn.metrics import classification_report

pred_lr = lr.predict(test_vectorx)
print("--- REPORTE DE CLASIFICACIÓN: REGRESIÓN LOGÍSTICA ---")
print(classification_report(test_y, pred_lr))

#Matriz de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_y, pred_lr)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()


#PASO FINAL --- Optimizacion del modelo

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear'], 'class_weight': [None, 'balanced'] 
}
svc_grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1)
svc_grid.fit(entr_vectorx, entr_y)

print(f"Mejores parámetros: {svc_grid.best_params_}")
print(f"Mejor score obtenido en validación: {svc_grid.best_score_:.4f}")

#Guardado
import joblib
joblib.dump(lr, "models/modelo_sentimientos.pkl")
joblib.dump(tf, "models/vectorizador.pkl")



