# VisualAnalyticsFinalLab

## 🎓 Visual Analytics Final Project: Alerta Temprana de Deserción Estudiantil

El proyecto integrará **Tableau**, **Streamlit**, **Machine Learning (ML)** y **Explainable AI (XAI)** para analizar y predecir el riesgo de deserción en la educación superior.

---

## 1. Project Report 

### 1.1. Problem Statement

La deserción estudiantil representa una pérdida significativa de recursos financieros y humanos, tanto para los estudiantes como para las instituciones. El problema a abordar es la **falta de mecanismos predictivos y explicativos** que permitan a las universidades identificar a los estudiantes en alto riesgo de deserción **al inicio de sus estudios** y entender **qué factores específicos** están impulsando ese riesgo.

### 1.2. Dataset Overview

* **Dataset:** Student Dropout and Academic Success (UCI Machine Learning Repository).
* **Fuente:** Universidad de Oporto, datos anonimizados de estudiantes matriculados en diferentes cursos de grado.
* **Análisis Exploratorio Básico:**
    * **Variables Clave:** Resultado final (Target: Deserción, Matriculado, Graduado), Ingreso familiar, Nivel educativo de los padres, Notas de admisión, Nacionalidad, Estado civil.
    * **Estadísticas de Interés:** Se analizará la tasa de deserción general, y se observará la distribución de la variable objetivo por Ingreso Familiar (ej., **Correlación:** ¿Los estudiantes de ingresos más bajos presentan una tasa de deserción significativamente mayor?).

### 1.3. Business Questions y Objetivos

| Tipo | Pregunta de Negocio (Analítica) | Objetivo del Proyecto |
| :--- | :--- | :--- |
| **Descriptivo (Tableau)** | ¿Cómo se distribuye la **tasa de deserción** entre los diferentes **programas de estudio** y grupos demográficos (ej. género, nacionalidad)? | **Visualizar** dónde se concentra el problema para asignar recursos de manera eficiente. |
| **Predictivo (ML)** | ¿Es posible predecir el **resultado final** del estudiante (Deserción/Graduación) con un alto grado de precisión utilizando variables socioeconómicas y académicas iniciales? | **Construir** un modelo de clasificación robusto ($\text{Accuracy} > 75\%$). |
| **Explicativo (XAI)** | ¿Qué **factores específicos** (ej. bajo rendimiento en el primer semestre, ingreso familiar) son los **más influyentes** para la predicción de deserción de un estudiante en particular? | **Explicar** los resultados del modelo, transformando la predicción en una recomendación accionable. |

### 1.4. Methodology (Metodología)

1.  **Limpieza y Preprocesamiento:** Codificación de variables categóricas (One-Hot Encoding) y normalización de variables numéricas.
2.  **Modelado (ML):** Uso de un algoritmo de **Clasificación Multiclase** (ej., Random Forest o XGBoost) para predecir la variable `Resultado final` (`Dropout`, `Enrolled`, `Graduate`).
3.  **Explicabilidad (XAI):** Aplicación de la librería **SHAP (SHapley Additive exPlanations)** para obtener valores de importancia de las *features* a nivel global y a nivel de instancia.
4.  **Visualización:** Creación de *storytelling* en Tableau para el análisis exploratorio y desarrollo de la aplicación interactiva en Streamlit.

---

## 2. Executable Project (Componentes Mínimos)

Se desarrollarán los cuatro componentes clave:

### 2.1. Tableau Storytelling

* **Propósito:** Análisis exploratorio y establecimiento de la narrativa.
* **Dashboard 1: Panorama Global:** Mapa de árbol mostrando las **tasas de deserción por Programa/Curso** de estudio.
* **Dashboard 2: El Factor Socioeconómico:** Gráficos de barras que muestren la relación entre el **Ingreso Familiar del Estudiante** y la tasa de deserción.
* **Dashboard 3: Las Señales de Alerta:** Gráfico de dispersión o *box plots* comparando las **Calificaciones de Ingreso o Primer Semestre** entre los estudiantes que desertaron vs. los que se graduaron.

### 2.2. Trained ML Model (Modelo Entrenado)

* **Tipo:** Modelo de **Clasificación Multiclase**.
* **Requisitos:** El modelo debe ser serializado (guardado en un archivo `.pkl` o similar) para ser cargado y utilizado dentro de la aplicación Streamlit.

### 2.3. Explainability using XAI (Explicabilidad)

* **Integración:** La lógica XAI debe estar integrada en Python (con la librería SHAP) y sus resultados deben ser visualizados en Streamlit.
* **Visualización Clave:** Un **gráfico de cascada (waterfall plot) SHAP** que muestre cómo cada variable de un estudiante específico (ej. "Ingreso Bajo", "Nota Media de 12") empuja la predicción hacia "Deserción" o "Graduado". 

### 2.4. Streamlit Web App

* **Propósito:** Interfaz interactiva para la intervención.
* **Estructura:**
    * **Sección 1: Entrada de Datos:** Widgets (*sliders*, cajas de texto) para que el usuario ingrese o modifique los datos de un estudiante hipotético o real.
    * **Sección 2: Predicción ML:** Muestra el resultado de la predicción (ej. "Riesgo Alto de Deserción: 78% de Probabilidad").
    * **Sección 3: La Explicación (XAI):** Muestra el **gráfico SHAP** para esa predicción específica. Esto permite al consejero académico ver **por qué** el modelo dio ese resultado.
    * **Funcionalidad Clave:** El consejero puede modificar una variable (ej. subir la beca) y ver inmediatamente **cómo cambia la predicción y la explicación**.

---

## 3. Presentation (10 Minutos)

La presentación debe seguir la narrativa del *storytelling*:

1.  **Introducción (1 min):** Definición del problema de la deserción y la importancia de la intervención temprana.
2.  **Análisis (2-3 min):** Presentación de los *dashboards* de Tableau (Mapa de Deserción, Perfil de Riesgo).
3.  **La Solución ML/XAI (3 min):** Explicación del modelo y de la **necesidad de XAI** (no solo predecir, sino explicar).
4.  **Demostración Streamlit (2-3 min):** Demostración interactiva del simulador de riesgo, mostrando cómo una variable cambia la predicción y la explicación.
5.  **Conclusiones y Futuro (1 min):** Resumen de los *insights* y siguientes pasos.
