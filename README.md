# svc-sentiment-optimizer
IMDB movie review classifier achieving 87% accuracy using SVM and TF-IDF vectorization.

Este proyecto implementa un sistema de aprendizaje automático (Machine Learning) para clasificar automáticamente el sentimiento de reseñas de películas (positivo/negativo). El enfoque no se limita a obtener un resultado, sino a seguir un ciclo de vida completo de Ciencia de Datos: limpieza, balanceo, comparación de modelos, optimización y serialización.

## 🚀 Metodología
El proyecto sigue un pipeline estándar de procesamiento de lenguaje natural (NLP):
1. **Preprocesamiento y Balanceo:** Uso de `RandomUnderSampler` para corregir el desequilibrio de clases en los datos originales.
2. **Vectorización:** Conversión de texto a representaciones numéricas mediante `TF-IDF` (ignorando *stop words*).
3. **Comparativa de Modelos:** Evaluación técnica de cuatro algoritmos (SVC, Naive Bayes, Regresión Logística y Árboles de Decisión).
4. **Optimización:** Ajuste de hiperparámetros mediante `GridSearchCV` con validación cruzada (`cv=5`) para maximizar el rendimiento.
5. **Evaluación:** Uso de `classification_report` y `Matriz de Confusión` para medir Precisión, Recall y F1-Score.

## 📊 Resultados Principales
Tras la optimización, el modelo de **Regresión Logística** alcanzó un **87% de F1-score** y **accuracy**, demostrando una gran estabilidad y equilibrio entre clases.

| Métrica | Valor |
| :--- | :--- |
| **Accuracy Global** | 87% |
| **F1-Score (Negativo)** | 0.86 |
| **F1-Score (Positivo)** | 0.87 |

## 🛠️ Stack Tecnológico
- **Python 3.x**
- **Scikit-learn:** Para el modelado, optimización y métricas.
- **Pandas:** Manipulación de datos.
- **Joblib:** Serialización y despliegue del modelo.
- **Matplotlib/Seaborn:** Visualización de resultados.

## 📁 Cómo ejecutar este proyecto
1. Instala las dependencias:
   ```bash
   pip install pandas scikit-learn joblib matplotlib seaborn
