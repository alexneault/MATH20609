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
    None

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
    
    wb.save()

    print(f"Les données ont été ajoutées à l'onglet '{output}' du fichier '{file_name}'.")


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)
    handle_inputs(sys.argv[1])
