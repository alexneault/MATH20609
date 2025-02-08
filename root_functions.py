# à faire :
# insérer les gif dans le excel
# secante
# newton
# quasi newton
# muller
#hello world

#!/usr/bin/env python
from csv import excel
from math import *
import sys
import matplotlib.axes
import pandas as pd
import numpy as np
import unicodedata
import xlwings as xw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from PIL import Image, ImageSequence
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.reader.excel import load_workbook

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



def add_animated_graph (approximations, inputs, func, name):
    #creates an animated graph
    values = list(approximations.values()) #x values of the function
    keys = list(approximations.keys()) #y values of the function
    fig, ax = plt.subplots() #initiate graph process
    x = np.linspace(inputs['min'][0],inputs['max'][0],1000) #graph the function through 1000 points between max and min
    y = [eval(func, globals(), {'x':i}) for i in x]
    plt.plot(x, y, label=f"f(x) = {func}", color="blue")
    colors = np.linspace(0, 1, len(approximations.keys()))
    scatter = plt.scatter(list(approximations.keys()),list(approximations.values()), c=colors, cmap="Greens",edgecolors='black', s=100, alpha=0.9)
    red_point = plt.scatter(keys[-1], values[-1], color='red', s=25, zorder=3, edgecolors="black")
    def update(frame): #creates an iteration to generate the gif graph
        scatter.set_offsets(np.c_[keys[:frame + 1], values[:frame + 1]])
        scatter.set_array(colors[:frame + 1])
        return scatter, red_point

    anim = FuncAnimation(fig, update, frames=len(keys) + 10, interval=5000 / len(keys), blit=True)  # generate the gif
    plt.legend()
    plt.colorbar(scatter)
    anim.save(f'{name} animation.gif', writer='pillow', fps=10) # save the gif in the folder

def merge_gifs_side_by_side(gif1_path, gif2_path, output_path):
    # Load both GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # Ensure both GIFs have the same number of frames
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]

    # Resize frames to have the same height
    height = min(frames1[0].height, frames2[0].height)
    frames1 = [f.resize((int(f.width * height / f.height), height)) for f in frames1]
    frames2 = [f.resize((int(f.width * height / f.height), height)) for f in frames2]

    # Merge frames side by side
    merged_frames = []
    for f1, f2 in zip(frames1, frames2):
        new_width = f1.width + f2.width
        new_frame = Image.new("RGBA", (new_width, height))
        new_frame.paste(f1, (0, 0))
        new_frame.paste(f2, (f1.width, 0))
        merged_frames.append(new_frame)

    # Save as a new GIF
    merged_frames[0].save(output_path, save_all=True, append_images=merged_frames[1:], loop=0, duration=gif1.info['duration'])


def bissection(inputs: dict, ws: xw.Sheet, min, max):
    try:
        bissection_approxs = {}
        col_bissection = inputs['bissection'][2]
        func = inputs['fonction'][0]
        precision_required = inputs['precision'][0]

        x1 = inputs['min'][0]
        x2 = inputs['max'][0]

        x1_context = {"x": x1}
        x1_result = eval(func, globals(), x1_context)
        x2_context = {"x": x2}
        x2_result = eval(func, globals(), x2_context)
        bissection_approxs[x1] = x1_result
        bissection_approxs[x2] = x2_result
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
                bissection_approxs[x3] = x3_result
                if x3_result > 0:
                    x2 = x3
                else:
                    x1 = x3
                bissection_list.append(x3)
                bissection_result = x3
                precision_result = x2-x1
                if len(bissection_approxs) > 2500:
                    bissection_result = "Aucun zero sur cette section"  # Cette section ne marche pas vraiment, si les deux points sont du même signe, aucune
                    proccess_failed = True
                    break
    except OverflowError:
        bissection_result = "Erreur, résultat des bornes trop élevé"
    except ZeroDivisionError:
        bissection_result = "Processus implique une division par 0"

    ws.range(f"C{col_bissection}").value = bissection_result
    populate_graph_data(inputs, "bissection", bissection_approxs, bissection_result)
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(bissection_approxs, inputs, func,'bissection')

def secante(inputs: dict, ws: xw.Sheet, min, max):
    None
#work in progress, faut je l'intègre dans notre gros script et rajouter safeguards.
     # values
    secante_list = []
    approxs = {}
    col_secante = inputs['secante'][2]
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

    precision_result=abs(x2-x1)

    while abs(precision_result) > abs(precision_required):
        fx1 = eval(func, globals(), {"x": x1})
        fx2 = eval(func, globals(), {"x": x2})

        if fx2 - fx1 ==0:
            secante_result = "Erreur : Division par Zero"
            break

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

    pointfixe_approxs = {}
    col_pointfixe = inputs['pointfixe'][2]
    func = inputs['fonction'][0]
    precision_required = inputs['precision'][0]

    x = sp.Symbol("x")
    func2 = sp.sympify(func)
    x1 = inputs['min'][0]
    x2 = inputs['max'][0]
    funcprime = sp.diff(func2,x)
    min_funcprime = funcprime.subs(x,x1)
    max_funcprime = funcprime.subs(x,x2)
    if abs(float(min_funcprime)) < 1 or abs(float(max_funcprime)) < 1:
        if float(min_funcprime) <= float(max_funcprime):
            x0 = x1
        else:
            x0 = x2
        bissectrice = x
        for i in range(1,250):
            pointfixe_approxs[x0] = func2.subs(x,x0)
            temp = func2.subs(x,x0)
            x0 = bissectrice.subs(x,temp)
        pointfixe_result = x0
        ws.range(f"C{col_pointfixe}").value = pointfixe_result
        add_animated_graph(pointfixe_approxs, inputs, func, 'pointfixe')

    else:
        pointfixe_result = "les valeurs absolues des dérivés au point max et min sont supérieurs ou égales à 1"
        ws.range(f"C{col_pointfixe}").value = pointfixe_result


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

    #merge_gifs_side_by_side('bissection animation.gif','pointfixe animation.gif', 'merged.gif')
    #add the animated graph je vais creer une loop pour le faire pour chacun des graphs plus tard.
    if input_data['animationordinateur'][0] == 1:
        outputsheet = wb.sheets['Output']
        outputsheet.pictures.add('bissection animation.gif', top=outputsheet.range("S5").top, left=sheet.range("S5").left)



    print(f"Les données ont été ajoutées à l'onglet '{output}' du fichier '{file_name}'.")
    #print(input_data)


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)
    handle_inputs(sys.argv[1])
