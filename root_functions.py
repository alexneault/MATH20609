#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import unicodedata
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

# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    inputs = xlsx_to_dict(file_name)

    # inputs.update({'x': 1}) # Only used for Test
    # print(eval_with_imports(inputs['fonction'], inputs)) # Test

    run_functions(inputs)
