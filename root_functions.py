#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import unicodedata
import xlwings as xw
from numpy import exp

# Ex: quasi-Newton -> quasinewton 
# Ex: Sécante -> secante 
def remove_accent_and_lowercase(input_string):
    nfkd_form = unicodedata.normalize('NFKD', input_string)
    return ''.join([char.lower() for char in nfkd_form if unicodedata.category(char) != 'Mn' and char.isalpha()])

# Excel -> dict/map
def xlsx_to_dict(file_name):
    result = {}
    try:
        df = pd.read_excel(file_name)

        if df.shape[1] >= 2:
            for _, row in df.iterrows():
                key = remove_accent_and_lowercase(str(row.iloc[0]))
                value = row.iloc[1]
                result[key] = value
        else:
            print("The file does not have enough columns.")
            return {}

    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return result

# Ajouter les imports qui nous faut pour eval('fonction')
def eval_with_imports(expression, custom_globals):
    allowed_globals = {
        'import': __import__,
        'math': __import__('math'),
        'np': __import__('numpy'),
        'exp': __import__('numpy').exp
    }

    if custom_globals:
        allowed_globals.update(custom_globals)

    return eval(expression, allowed_globals)

# Check if functions are enabled (== 1) and run them
def run_functions(inputs: dict):
    if inputs['bissection'] == 1: bissection(inputs)
    if inputs['secante'] == 1: secante(inputs)
    if inputs['newton'] == 1: newton(inputs)
    if inputs['quasinewton'] == 1: quasi_newton(inputs)
    if inputs['muller'] == 1: muller(inputs)
    if inputs['pointfixe'] == 1: pointfixe(inputs)

def bissection(inputs):
    None

def secante(inputs):
    None

def newton(inputs):
    None

def quasi_newton(inputs):
    None

def muller(inputs):
    None

def pointfixe(inputs):
    None

def copy_inputs(file_name):
    # Ouvrir le fichier Excel
    wb = xw.Book(file_name)
    sheet = wb.sheets["Inputs"]

    # Lire une cellule ou une plage
    fonctions_col = sheet.range("A3:A100").value  # Lire colonne A
    binaire_col = sheet.range("B3:B100").value  # Lire colonne B

    functions = zip(fonctions_col, binaire_col)  # créer une liste avec les fonctions

    functions_filtered = [item for item in functions if item != (None, None)]  # retirer les valeurs vides

    # Insérer les inputs originaux dans la feuille output de notre excel
    nom_onglet = "Output"
    nom_fichier = file_name

    try:
        ws = wb.sheets.add(name=nom_onglet)
    except ValueError:
        ws = wb.sheets[nom_onglet]

    for i, (nom_fonction, variable_binaire) in enumerate(functions_filtered, start=2):  # Commencer à la ligne 1
        ws.range(f"A{i}").value = nom_fonction
        ws.range(f"B{i}").value = variable_binaire

    ws.range("A1").value = "Output"  # première valeur dans la case A1

    wb.save()

    print(f"Les données ont été ajoutées à l'onglet '{nom_onglet}' du fichier '{nom_fichier}'.")


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    inputs = xlsx_to_dict(file_name)
    copy_inputs("Devoir1_Entame.xlsm")
    # inputs.update({'x': 1}) # Only used for Test
    # print(eval_with_imports(inputs['fonction'], inputs)) # Test
    run_functions(inputs)
