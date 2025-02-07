#!/usr/bin/env python

from math import *
import sys
import matplotlib.axes
import pandas as pd
import numpy as np
import unicodedata
import xlwings as xw
import matplotlib.pyplot as plt

# Globals
nb_iterations = 10
approxs_plot_data = {}
root_plot_data = {}

colors = "gbrcmykw"

# Ex: quasi-Newton -> quasinewton 
# Ex: Sécante -> secante 
def remove_accent_and_lowercase(input_string):
    nfkd_form = unicodedata.normalize('NFKD', input_string)
    return ''.join([char.lower() for char in nfkd_form if unicodedata.category(char) != 'Mn' and char.isalpha()])

def run_functions(inputs: dict, ws: xw.Sheet, min, max):
    if inputs['bissection'][0] == 1: bissection(inputs, ws, min, max)
    if inputs['secante'][0] == 1: secante(inputs, ws, min, max)
    if inputs['newton'][0] == 1: newton(inputs, ws, min, max)
    if inputs['quasinewton'][0] == 1: quasi_newton(inputs, ws, min, max)
    if inputs['muller'][0] == 1: muller(inputs, ws, min, max)
    if inputs['pointfixe'][0] == 1: pointfixe(inputs, ws, min, max)

def bissection(inputs: dict, ws: xw.Sheet, min, max):
    try:
        approxs = {}
        col_bissection = inputs['bissection'][2]
        func = inputs['fonction'][0]
        precision_required = inputs['precision'][0]

        x1 = inputs['min'][0]
        x2 = inputs['max'][0]

        x1_context = {"x": x1}
        x1_result = eval(func, globals(), x1_context)
        x2_context = {"x": x2}
        x2_result = eval(func, globals(), x2_context)
        approxs[x1] = x1_result
        approxs[x2] = x2_result
        if x1_result > x2_result:
            x1 = inputs['max'][0]
            x2 = inputs['min'][0]

        bissection_list = []

        if x1_result * x2_result > 0:
            bissection_result = "Aucun zero sur cette section" # Cette section ne marche pas vraiment, si les deux points sont du même signe, aucune réponse n'est retournée
        else:
            precision_result = 1
            while precision_result > precision_required:
                x3 = (x1 + x2)/2
                x3_context = {"x": x3}
                x3_result = eval(func, globals(), x3_context)
                approxs[x3] = x3_result
                if x3_result > 0:
                    x2 = x3
                else:
                    x1 = x3
                bissection_list.append(x3)
                bissection_result = x3
                precision_result = x2-x1
    except OverflowError:
        bissection_result = "Erreur, résultat des bornes trop élevé"
    ws.range(f"C{col_bissection}").value = bissection_result
    populate_graph_data(inputs, "bissection", approxs, bissection_result)

def secante(inputs: dict, ws: xw.Sheet, min, max):
#work in progress, faut je l'intègre dans notre gros script et rajouter safeguards.
    # values
    try:
        approxs = {}
        col_secante = inputs["secante"][2]
        func = inputs["function"][0]
        precision_required = inputs['precision'][0]

        x1 = inputs['min'][0]
        x2 = inputs['max'][0]

        x1_context = {"x": x1}
        x1_result = eval(func, globals(), x1_context)
        x2_context = {"x": x2}
        x2_result = eval(func, globals(), x2_context)
        approxs[x1] = x1_result
        approxs[x2] = x2_result
        
        secante_list = []
        precision_result=abs(x2-x1)
        
            while abs(precision_result) > abs(precision_required):
                fx1 = eval(func, globals(), {"x": x1})
                fx2 = eval(func, globals(), {"x": x2})
                
                x3 = x2-(fx2/((fx2-fx1)/(x2-x1)))
                x3_context = {"x": x3}
                x3_result = eval(func, globals(), x3_context)
                approxs[x3] = x3_result
                x1, x2 = x2, x3
                precision_result = abs(x2-x1)
                secante_list.append(x3)
                secante_result= x3

    ws.range(f"C{col_secante}").value = secante_result
    populate_graph_data(inputs, "secante", approxs, secante_result)

def newton(inputs: dict, ws: xw.Sheet, min, max):
    None

def quasi_newton(inputs: dict, ws: xw.Sheet, min, max):
    None

def muller(inputs: dict, ws: xw.Sheet, min, max):
    None

def pointfixe(inputs: dict, ws: xw.Sheet, min, max):
    None

def initiate_plot(nb_of_plot: int):
    if(nb_of_plot == 1):
        return plt.subplots(figsize=(6, 4))
    return plt.subplots(1, nb_of_plot, figsize=(12, 4))

def add_root_plot(axs: matplotlib.axes, nb_of_plot):
    plot = axs if nb_of_plot == 1 else axs[0]
    df = pd.DataFrame(root_plot_data.items(), columns=['funct','root'])
    plot.bar(df['funct'], df['root'], color='blue')
    plot.set_title('Racines')
    plot.set_xlabel('Function')
    plot.set_ylabel('Root value')
    plot.legend()

def add_approx_plot(axs: matplotlib.axes, nb_of_plot):
    plot = axs if nb_of_plot == 1 else axs[1]
    color_index = 0
    for funct, data in approxs_plot_data.items():
        df = pd.DataFrame(data.items(), columns=['x','value'])
        plot.scatter(df['x'], df['value'], label=funct, color=colors[color_index])
        color_index += 1
    plot.set_title('Approximations')
    plot.set_xlabel('x')
    plot.set_ylabel('f(x)')
    plot.legend()

def populate_graph_data(inputs: str, funct: str, approxs: dict, res):
    if inputs['graphiqueracine'][0] == 1:
        root_plot_data[funct] = res
    if inputs['graphiqueapproximations'][0] == 1:
        approxs_plot_data[funct] = approxs


        

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
    
    nb_of_plot = 0
    nb_of_plot = int(input_data['graphiqueracine'][0] + input_data['graphiqueapproximations'][0])
    if nb_of_plot > 0:
        fig, axes = initiate_plot(nb_of_plot)
        if input_data['graphiqueracine'][0] == 1:
            add_root_plot(axes, nb_of_plot)
        if input_data['graphiqueapproximations'][0] == 1:
            add_approx_plot(axes, nb_of_plot)
        ws.pictures.add(fig, name='Graphiques', update=True, left=ws.range('E8').left, top=ws.range('E8').top)

    print(f"Les données ont été ajoutées à l'onglet '{output}' du fichier '{file_name}'.")
    #print(input_data)


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)
    handle_inputs(sys.argv[1])
