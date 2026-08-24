# 🚀 Proyecto de Ingeniería de Datos y Machine Learning: Detección de Fraude Financiero con Apache Spark (PySpark)

## 📌 Resumen Ejecutivo del Proyecto

Este proyecto implementa un flujo de trabajo analítico y predictivo de extremo a extremo para la **detección de transacciones financieras fraudulentas** utilizando **Apache Spark (PySpark)** y **MLlib**. El diseño metodológico prioriza la robustez en entornos de Big Data, abarcando desde la ingesta y el preprocesamiento masivo hasta la mitigación de sesgos, control de multicolinealidad, entrenamiento distribuido con balanceo de clases y evaluación mediante métricas estándar de la industria (Curva ROC, AUC y Matriz de Confusión).

---

## 🛠️ Fases del Flujo de Trabajo Desarrollado

1. **Limpieza y Preparación Inicial:**
* Eliminación de columnas identificadoras redundantes (terminadas en `_id`) para evitar ruido analítico en los modelos predictivos.


* Aplicación de **transformación cíclica** (seno y coseno) sobre la variable `hour` para modelar correctamente la periodicidad temporal de 24 horas.




2. **Ingeniería de Características y Tratamiento de Asimetría (`Skewness`):**
* Evaluación estadística del sesgo a la derecha en variables continuas.


* Aplicación de la transformación logarítmica segura (`log1p` o $\log(x+1)$) para estabilizar la varianza y prevenir indeterminaciones matemáticas al lidiar con valores en cero.




3. **Diagnóstico y Control de Multicolinealidad (VIF):**
* Conversión de variables categóricas mediante `StringIndexer` y `OneHotEncoder` (configurando `dropLast=True` para evitar la "Trampa de las variables ficticias").


* Implementación de un bloque de diagnóstico de multicolinealidad basado en la matriz inversa de correlación de Pearson para calcular el **Factor de Inflación de la Varianza (VIF)**. Esto permitió identificar y descartar la variable `country`, la cual presentaba un VIF crítico ($>18$), eliminando la redundancia lineal.




4. **Mitigación de Fuga de Datos (*Data Leakage*) y Escalamiento en Pipeline:**
* Consolidación de todas las transformaciones de preprocesamiento y modelado dentro de un `Pipeline` unificado de Spark ML, asegurando que las divisiones de entrenamiento (*train*) y prueba (*test*) mantengan el aislamiento estadístico.


* Uso del `StandardScaler` (con `withMean=False` para preservar la dispersión vectorial).




5. **Mitigación de Clases Desbalanceadas:**
* Dado que el dataset presentaba un fuerte desbalance (aproximadamente un 5% de transacciones fraudulentas), se implementó una estrategia de **pesos por clase (`class_weight`)** calculada dinámicamente, incrementando la penalización matemática ante la omisión de fraudes.




6. **Entrenamiento y Evaluación Profesional:**
* Entrenamiento de una **Regresión Logística** distribuida.


* Análisis iterativo de los coeficientes del modelo para detectar fugas de datos (*data leakage*), decidiendo entrenar una versión optimizada sin variables proxy de riesgo artificial.
* Obtención de un **AUC-ROC de 0.9620** y una **Matriz de Confusión** detallada (logrando atrapar 16 de 17 fraudes reales).



---

## 💡 Interacción Colaborativa (Humano-IA) y Razonamiento Analítico

Durante el desarrollo del proyecto, la interacción con Gemini AI operó bajo un esquema riguroso de **copilotaje técnico y descomposición analítica**, destacando los siguientes hitos de razonamiento crítico por parte del autor:

* **Escepticismo Analítico ante Resultados Anómalos:** Tras obtener inicialmente un AUC perfecto de `1.0000`, el criterio del desarrollador cuestionó el resultado, infiriendo correctamente la existencia de un posible sobreajuste (*overfitting*) o fuga de datos. Al analizar los coeficientes del modelo junto con el copiloto, se descubrió que los *risk scores* (`ip_risk_score` y `device_risk_score`) dominaban con pesos masivos ($>6.0$), funcionando como una "variable proxy" del fraude. Se tomó la decisión analítica de reentrenar el modelo excluyéndolas para evaluar el verdadero valor predictivo de los patrones de comportamiento económico.
* **Comprensión Arquitectónica:** Se asimiló y validó la importancia de encapsular los estimadores y transformadores en un `Pipeline` para evitar contaminar el conjunto de validación con métricas del conjunto de entrenamiento (*data leakage*), un concepto fundamental en MLOps.



---

## 🔮 Futuras Mejoras Propuestas (Escalabilidad y Automatización MLOps)

Como propuesta de mejora y evolución del código actual hacia un entorno de producción automatizado, se plantea:

* **Automatización Dinámica de la Depuración por VIF:** Actualmente, la identificación y eliminación de columnas con multicolinealidad severa ($VIF > 5$) se realiza de forma manual tras inspeccionar la tabla de resultados. La mejora propuesta consiste en **desarrollar un bucle iterativo automatizado** que calcule matricialmente los valores VIF, identifique la variable que supere el umbral crítico, la excluya de la lista de características de manera programática, y re-ejecute el cálculo de forma recursiva hasta que todas las variables del sistema cumplan con el criterio de estabilidad estadística. Esto eliminará la intervención manual y permitirá escalar el pipeline a datasets con cientos de dimensiones categóricas y numéricas.