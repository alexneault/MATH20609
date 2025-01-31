#!/usr/bin/env python

from math import *
import sys
import pandas as pd
import numpy as np
import unicodedata
import xlwings as xw

# Globals
nb_iterations = 10

# Ex: quasi-Newton -> quasinewton 
# Ex: Sécante -> secante 
def remove_accent_and_lowercase(input_string):
    nfkd_form = unicodedata.normalize('NFKD', input_string)
    return ''.join([char.lower() for char in nfkd_form if unicodedata.category(char) != 'Mn' and char.isalpha()])

def ensure_equals_prefix(expression) -> str:
    if not expression.startswith('='):
        expression = '=' + expression
    return expression

def run_functions(inputs: dict, ws: xw.Sheet, min, max):
    if inputs['bissection'][0] == 1: bissection(inputs, ws, min, max)
    if inputs['secante'][0] == 1: secante(inputs, ws, min, max)
    if inputs['newton'][0] == 1: newton(inputs, ws, min, max)
    if inputs['quasinewton'][0] == 1: quasi_newton(inputs, ws, min, max)
    if inputs['muller'][0] == 1: muller(inputs, ws, min, max)
    if inputs['pointfixe'][0] == 1: pointfixe(inputs, ws, min, max)
    min_max(inputs, ws, min, max)


def min_max (inputs: dict, ws: xw.Sheet, min, max):
    col_min = inputs['min'][2]
    col_max = inputs['max'][2]
    func = inputs['fonction'][0]

    min_context = {"x": inputs['min'][0]}
    min_result = eval(func, globals(), min_context)

    max_context = {"x": inputs['max'][0]}
    max_result = eval(func, globals(), max_context)

    ws.range(f"C{col_min}").value = min_result
    ws.range(f"C{col_max}").value = max_result


def bissection(inputs: dict, ws: xw.Sheet, min, max):
    try:
        col_bissection = inputs['bissection'][2]
        func = inputs['fonction'][0]
        precision_required = inputs['precision'][0]

        x1 = inputs['min'][0]
        x2 = inputs['max'][0]

        x1_context = {"x": x1}
        x1_result = eval(func, globals(), x1_context)
        x2_context = {"x": x2}
        x2_result = eval(func, globals(), x2_context)
        if x1_result > x2_result:
            x1 = inputs['max'][0]
            x2 = inputs['min'][0]

        print(x1_result)
        print(x2_result)
        if x1_result * x2_result > 0:
            bissection_result = "Aucun zero sur cette section" # Cette section ne marche pas vraiment, si les deux points sont du même signe, aucune réponse n'est retournée
        else:
            precision_result = 1
            while precision_result > precision_required:
                x3 = (x1 + x2)/2
                x3_context = {"x": x3}
                x3_result = eval(func, globals(), x3_context)
                print(x3, x3_result)
                if x3_result > 0:
                    x2 = x3
                else:
                    x1 = x3
                bissection_result = x3
                precision_result = x2-x1
    except OverflowError:
        bissection_result = "Erreur, résultat des bornes trop élevé"

    print(bissection_result)
    ws.range(f"C{col_bissection}").value = bissection_result


def secante(inputs: dict, ws: xw.Sheet, min, max):
    None

def newton(inputs: dict, ws: xw.Sheet, min, max):
    None

def quasi_newton(inputs: dict, ws: xw.Sheet, min, max):
    None

def muller(inputs: dict, ws: xw.Sheet, min, max):
    None

def pointfixe(inputs: dict, ws: xw.Sheet, min, max):
    None

def handle_inputs(file_name: str):
    output = "Output"
    inputs = "Inputs"
    results = "Results"

    # Ouvrir le fichier Excel
    wb = xw.Book(file_name)
    sheet = wb.sheets["Inputs"]
    input_data = {}
    functions = zip(sheet.range("A3:A100").value, sheet.range("B3:B100").value)
    functions_filtered = [item for item in functions if item != (None, None)]  # retirer les valeurs vides

    # Insérer les inputs originaux dans la feuille output de notre excel
    try:
        ws = wb.sheets.add(name=output, after=inputs)
    except ValueError:
        ws = wb.sheets[output]
    
    for i, (nom_fonction, value) in enumerate(functions_filtered, start=2):  # Commencer à la ligne 1
        ws.range(f"A{i}").value = nom_fonction
        ws.range(f"B{i}").value = value
        input_data[remove_accent_and_lowercase(ws.range(f"A{i}").value)] = (value, nom_fonction, i)

    x_min = input_data['min']
    x_max = input_data['max'] 
    run_functions(input_data, ws, x_min, x_max)

    ws.range("A1").value = output
    ws.range("C1").value = results

    print(f"Les données ont été ajoutées à l'onglet '{output}' du fichier '{file_name}'.")
    print(input_data)


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)
    handle_inputs(sys.argv[1])
