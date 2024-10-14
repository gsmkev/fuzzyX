import pandas as pd
import re
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os
import nltk
from tabulate import tabulate

# Descargar y configurar VADER lexicon
nltk.download('vader_lexicon', download_dir=os.getcwd())
nltk.data.path.append(os.getcwd())
os.system('cls||clear')

# Iniciar medición de tiempo
start_time = time.time()

# Cargar los datos
traindata = pd.read_csv('input.csv')

# Módulo 1: Procesamiento del texto
def process_text_data(data):
    """
    Procesa los datos de texto eliminando caracteres especiales, URLs y reemplazando contracciones comunes.
    Args:
        data (pandas.DataFrame): DataFrame que contiene una columna 'sentence' con los textos a procesar.
    Returns:
        pandas.DataFrame: DataFrame con la columna 'sentence' procesada.
    """
    def clean(phrase):
        """
        Elimina caracteres especiales y URLs, y reemplaza contracciones comunes en una frase.
        Args:
            phrase (str): La frase que se va a procesar.
        Returns:
            str: La frase procesada con contracciones reemplazadas y caracteres especiales eliminados.
        """
        # Eliminar caracteres especiales y URLs
        phrase = re.sub(r"@", "", phrase)
        phrase = re.sub(r"http\S+", "", phrase)
        phrase = re.sub(r"#", "", phrase)
        
        # Reemplazar contracciones comunes
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        return phrase
    
    # Aplicar la función clean a la columna 'sentence' para eliminar caracteres especiales, URLs y reemplazar contracciones
    data['sentence'] = data['sentence'].apply(lambda x: clean(x.lower()))
    return data

# Módulo 2: Análisis de sentimiento usando VADER
def analyze_sentiment(data):
    """
    Analiza el sentimiento de las oraciones en el DataFrame dado.
    Args:
        data (DataFrame): Un DataFrame que contiene una columna 'sentence' con las oraciones a analizar.
    Returns:
        DataFrame: El DataFrame original con tres nuevas columnas: 'pos', 'neg', y 'neu', que representan
                   los puntajes de sentimiento positivo, negativo y neutral respectivamente.
    """
    # Inicializar el analizador de sentimientos VADER
    sid = SentimentIntensityAnalyzer()
    
    # Calcular los puntajes de sentimiento para cada oración
    data['pos'] = data['sentence'].apply(lambda x: sid.polarity_scores(x)['pos'])
    data['neg'] = data['sentence'].apply(lambda x: sid.polarity_scores(x)['neg'])
    data['neu'] = data['sentence'].apply(lambda x: sid.polarity_scores(x)['neu'])
    return data

# Módulo 3: Aplicar las reglas fuzzy
def fuzzification(pos_score, neg_score, _, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos):
    """
    Aplica reglas difusas para calcular una agregación de activaciones.
    Args:
        pos_score (float): Puntuación positiva.
        neg_score (float): Puntuación negativa.
        x_op (array): Rango de operación.
        p_lo (array): Membresía baja positiva.
        p_md (array): Membresía media positiva.
        p_hi (array): Membresía alta positiva.
        n_lo (array): Membresía baja negativa.
        n_md (array): Membresía media negativa.
        n_hi (array): Membresía alta negativa.
        op_neg (array): Operación negativa.
        op_neu (array): Operación neutral.
        op_pos (array): Operación positiva.
    Returns:
        array: Agregación de todas las activaciones.
    """
    # Interpolación de la membresía
    p_level_lo = fuzz.interp_membership(np.arange(0, 1, 0.1), p_lo, pos_score)
    p_level_md = fuzz.interp_membership(np.arange(0, 1, 0.1), p_md, pos_score)
    p_level_hi = fuzz.interp_membership(np.arange(0, 1, 0.1), p_hi, pos_score)
    
    n_level_lo = fuzz.interp_membership(np.arange(0, 1, 0.1), n_lo, neg_score)
    n_level_md = fuzz.interp_membership(np.arange(0, 1, 0.1), n_md, neg_score)
    n_level_hi = fuzz.interp_membership(np.arange(0, 1, 0.1), n_hi, neg_score)
    
    # Reglas difusas
    active_rule1 = np.fmin(p_level_lo, n_level_lo)
    active_rule2 = np.fmin(p_level_md, n_level_lo)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)
    active_rule4 = np.fmin(p_level_lo, n_level_md)
    active_rule5 = np.fmin(p_level_md, n_level_md)
    active_rule6 = np.fmin(p_level_hi, n_level_md)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)
    active_rule8 = np.fmin(p_level_md, n_level_hi)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)
    
    # Agregación de reglas
    n2 = np.fmax(active_rule4, np.fmax(active_rule7, active_rule8))
    op_activation_lo = np.fmin(n2, op_neg)

    neu2 = np.fmax(active_rule1, np.fmax(active_rule5, active_rule9))
    op_activation_md = np.fmin(neu2, op_neu)

    p2 = np.fmax(active_rule2, np.fmax(active_rule3, active_rule6))
    op_activation_hi = np.fmin(p2, op_pos)
    
    # Agregar todas las activaciones
    aggregated = np.fmax(op_activation_lo, np.fmax(op_activation_md, op_activation_hi))
    
    return aggregated

# Módulo 4: Definir las funciones de membresía fuzzy
def define_fuzzy_membership():
    """
    Define las funciones de membresía fuzzy para varias variables.
    Args:
        None
    Returns:
        x_p (numpy.ndarray): Rango de la variable p.
        x_n (numpy.ndarray): Rango de la variable n.
        x_op (numpy.ndarray): Rango de la variable op.
        p_lo (numpy.ndarray): Función de membresía fuzzy para p baja.
        p_md (numpy.ndarray): Función de membresía fuzzy para p media.
        p_hi (numpy.ndarray): Función de membresía fuzzy para p alta.
        n_lo (numpy.ndarray): Función de membresía fuzzy para n baja.
        n_md (numpy.ndarray): Función de membresía fuzzy para n media.
        n_hi (numpy.ndarray): Función de membresía fuzzy para n alta.
        op_neg (numpy.ndarray): Función de membresía fuzzy para op negativa.
        op_neu (numpy.ndarray): Función de membresía fuzzy para op neutral.
        op_pos (numpy.ndarray): Función de membresía fuzzy para op positiva.
    """
    # Rango de las variables
    x_p = np.arange(0, 1, 0.1)
    x_n = np.arange(0, 1, 0.1)
    x_op = np.arange(0, 10, 1)

    # Funciones de membresía fuzzy
    p_lo = fuzz.trimf(x_p, [0, 0, 0.5])
    p_md = fuzz.trimf(x_p, [0, 0.5, 1])
    p_hi = fuzz.trimf(x_p, [0.5, 1, 1])
    
    n_lo = fuzz.trimf(x_n, [0, 0, 0.5])
    n_md = fuzz.trimf(x_n, [0, 0.5, 1])
    n_hi = fuzz.trimf(x_n, [0.5, 1, 1])
    
    op_neg = fuzz.trimf(x_op, [0, 0, 5])
    op_neu = fuzz.trimf(x_op, [0, 5, 10])
    op_pos = fuzz.trimf(x_op, [5, 10, 10])
    
    return x_p, x_n, x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos

# Módulo 5: Defuzzificación y clasificación
def defuzzify_and_classify(aggregated, x_op):
    """
    Desfuzzifica y clasifica el resultado basado en la salida desfuzzificada.
    Args:
        aggregated (array-like): El conjunto de datos agregados.
        x_op (array-like): El conjunto de valores x correspondientes.
    Returns:
        tuple: Una tupla que contiene la clasificación (str) y el valor desfuzzificado (float).
    """
    defuzzified_output = fuzz.defuzz(x_op, aggregated, 'centroid')
    if 0 <= defuzzified_output < 3.33:
        classification = "Negative"
    elif 3.33 <= defuzzified_output <= 6.66:
        classification = "Neutral"
    else:
        classification = "Positive"
    
    return classification, defuzzified_output

# Módulo 6: Generar un resumen de los resultados
def benchmarks_and_summary(df, total_time, total_fuzzy_time, total_defuzz_time):
    """
    Genera un resumen de los resultados del análisis de sentimiento y tiempos de ejecución.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados del análisis de sentimiento.
        total_time (float): Tiempo total de ejecución.
        total_fuzzy_time (float): Tiempo total de ejecución de las reglas fuzzy.
        total_defuzz_time (float): Tiempo total de ejecución de la defuzzificación.
    
    Returns:
        None
    """
    # Crear el resumen
    summary_data = {
        "Clasificación": ["Positivos", "Negativos", "Neutrales", "Total Procesado", "Tiempo Total de Ejecución (s)", "Tiempo Total Fuzzy (s)", "Tiempo Total Defuzz (s)"],
        "Cantidad": [
            (df["Clasificación"] == "Positive").sum(),
            (df["Clasificación"] == "Negative").sum(),
            (df["Clasificación"] == "Neutral").sum(),
            len(df),
            f"{total_time:.2f}",
            f"{total_fuzzy_time:.2f}",
            f"{total_defuzz_time:.2f}"
        ]
    }
    
    # Crear un DataFrame para el resumen
    summary_df = pd.DataFrame(summary_data)
    
    # Imprimir el resumen usando tabulate
    print("\nResumen de los tweets:")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False, colalign=("left", "center")))

# Auxiliar: Visualización de las funciones de membresía y el resultado defuzzificado
def visualize_memberships(aggregated, x_op, defuzzified_output, op_neg, op_neu, op_pos):
    """
    Visualiza las funciones de membresía difusas y el resultado defuzzificado.

    Args:
        aggregated (array-like): Salida agregada de las funciones de membresía.
        x_op (array-like): Valores del eje x para las funciones de membresía.
        defuzzified_output (float): Valor defuzzificado.
        op_neg (array-like): Valores de membresía para la función negativa.
        op_neu (array-like): Valores de membresía para la función neutral.
        op_pos (array-like): Valores de membresía para la función positiva.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_op, op_neg, 'b', linestyle='--', label='Negative')
    plt.plot(x_op, op_neu, 'g', linestyle='--', label='Neutral')
    plt.plot(x_op, op_pos, 'r', linestyle='--', label='Positive')
    plt.fill_between(x_op, np.zeros_like(x_op), aggregated, facecolor='orange', alpha=0.7, label='Aggregated Output')
    plt.axvline(defuzzified_output, color='red', linestyle='--', label=f'COA = {defuzzified_output:.2f}')
    plt.title('Fuzzy Membership Functions and Defuzzified Output')
    plt.xlabel('Score')
    plt.ylabel('Membership')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función principal
def main():
    # Módulo 1: Procesamiento del texto según las instrucciones de la Sección 3.1.
    processed_data = process_text_data(traindata)
    
    # Módulo 2: Análisis de sentimiento usando VADER según las instrucciones de la Sección 3.2.
    analyzed_data = analyze_sentiment(processed_data)
    
    # Módulo 4: Definir las funciones de membresía fuzzy según las instrucciones de la Sección 3.3.2 y 3.3.3.
    _, _, x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos = define_fuzzy_membership()
    
    # Crear una lista para almacenar los datos de la tabla
    data = []
    
    # Iterar sobre los tweets para aplicar las reglas fuzzy y la defuzzificación
    for _, row in analyzed_data.iterrows():
        pos_score = row['pos']
        neg_score = row['neg']
        
        # Módulo 3: Aplicar la fuzzificación según las instrucciones de la Sección 3.3.1.
        time_fuzzy_start = time.time()
        aggregated = fuzzification(pos_score, neg_score, x_op, p_lo, p_md, p_hi, n_lo, n_md, n_hi, op_neg, op_neu, op_pos)
        time_fuzzy_end = time.time()
        
        # Módulo 5: Defuzzificación y clasificación según las instrucciones de la Sección 3.3.4.
        time_defuzz_start = time.time()
        classification, defuzzified_score = defuzzify_and_classify(aggregated, x_op)
        time_defuzz_end = time.time()
        
        # Añadir los datos a la lista
        data.append([
            row['sentence'],
            classification,
            pos_score,
            neg_score,
            row['neu'],
            defuzzified_score,
            time_fuzzy_end - time_fuzzy_start,
            time_defuzz_end - time_defuzz_start
        ])
        
        # Visualización de las membresías y resultados
        #visualize_memberships(aggregated, x_op, defuzzified_score, op_neg, op_neu, op_pos)
        
    # Crear un DataFrame con los datos
    df = pd.DataFrame(data, columns=["Tweet", "Clasificación", "Positivo", "Negativo", "Neutral", "Puntuación Defuzzificada", "Tiempo Fuzzy", "Tiempo Defuzz"])
    
    # Calcular el tiempo total de ejecución
    total_time = time.time() - start_time
    total_fuzzy_time = df["Tiempo Fuzzy"].sum()
    total_defuzz_time = df["Tiempo Defuzz"].sum()
    
    # Imprimir la tabla usando tabulate con alineación personalizada
    df = df.drop(columns=["Tiempo Fuzzy", "Tiempo Defuzz"])
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('output.csv', index=False)
    
    print("Resultados del análisis de sentimiento (head):")
    print(tabulate(df.head(), headers="keys", tablefmt="grid", showindex=False, colalign=("left", "center", "center", "center", "center")))

    # Módulo 6: Generar un resumen de los resultados
    benchmarks_and_summary(df, total_time, total_fuzzy_time, total_defuzz_time)

# Ejecutar la función principal
if __name__ == "__main__":#
    main()