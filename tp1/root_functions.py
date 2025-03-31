# à faire :
# insérer les gif dans le excel
# secante
# newton
# quasi newton
# muller
#hello world
#test



#!/usr/bin/env python
from csv import excel
from math import *
import sys
from time import process_time, process_time_ns
import matplotlib.axes
import pandas as pd
import numpy as np
import unicodedata
import xlwings as xw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
#import imageio
from PIL import Image, ImageSequence
import os
from sympy import Abs

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

#nbr max. d'itérations à effectuer
wb = xw.Book("Devoir1_Entame.xlsm")
ws= wb.sheets[0]
iterations = ws.range("B23").value

def add_animated_graph (approximations, inputs, func, name):
    #creates an animated graph
    script_dir = os.path.dirname(__file__)  # folder where this .py file lives
    gif_path = os.path.join(script_dir, f"{name}_animation.gif")
    values = list(approximations.values()) #x values of the function
    keys = list(approximations.keys()) #y values of the function
    fig, ax = plt.subplots() #initiate graph process
    x = np.linspace(float(min(keys)),float(max(keys)),1000) #graph the function through 1000 points between max and min
    y = [eval(func, globals(), {'x':i}) for i in x]
    plt.plot(x, y, label=f"f(x) = {func}", color="blue")
    colors = np.linspace(0, 1, len(approximations.keys()))
    scatter = plt.scatter(list(approximations.keys()),list(approximations.values()), c=colors, cmap="Greens",edgecolors='black', s=100, alpha=0.9)
    red_point = plt.scatter(keys[-1], values[-1], color='red', s=25, zorder=3, edgecolors="black")
    red_point.set_label("Dernière itération")
    ax.text(
        keys[-1],
        (max(y)-min(y))*0.05+values[-1],
        f"({keys[-1]:.3f}, {values[-1]:.3f})",
        bbox=dict(facecolor="white", alpha=0.8),
        ha="center",
        va="bottom")
    def update(frame): #creates an iteration to generate the gif graph
        scatter.set_offsets(np.c_[keys[:frame + 1], values[:frame + 1]])
        scatter.set_array(colors[:frame + 1])
        return scatter, red_point
    anim = FuncAnimation(fig, update, frames=len(keys) + 100, interval=5000000, blit=True)  # generate the gif
    ax.set_title(f"Itérations de {name}")
    ax.set_xlabel("Axe des X")
    ax.set_ylabel("Axe des Y")
    plt.legend()
    plt.colorbar(scatter)
    anim.save(gif_path, writer='pillow', fps=3) # save the gif in the folder

def merge_gifs_side_by_side(gif1_path, gif2_path, output_path): #fonction inutilisée, mais disponible
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
    #établir le temps maximal
    max_time = inputs['tempslimite'][0]
    t1_start = process_time() 
    try:
        #set up de la fonction, établir les paramètres de base
        bissection_approxs = {}
        col_bissection = inputs['bissection'][2]
        func = inputs['fonction'][0]
        precision_required = inputs['precision'][0]

        x1 = inputs['min'][0]
        x2 = inputs['max'][0]
        #Calculer la valeur de x1 et x2 selon la fonction
        x1_context = {"x": x1}
        x1_result = eval(func, globals(), x1_context)
        x2_context = {"x": x2}
        x2_result = eval(func, globals(), x2_context)
        bissection_approxs[x1] = x1_result
        bissection_approxs[x2] = x2_result

        ite = iterations
        countr = 0
        #trouver lequel de x1 ou x2 a le plus grand y. Le plus grand y détermine x2
        if x1_result > x2_result:
            x1 = inputs['max'][0]
            x2 = inputs['min'][0]

        bissection_list = []

        if x1_result * x2_result > 0:
            #si x1 et x2 sont de même signe, la méthode ne permettra pas de trouver une racine
            bissection_result = "La bissection ne permet pas d'identifier de racine sur cette section (deux résultats de même signe)" # Cette section ne marche pas vraiment, si les deux points sont du même signe, aucune réponse n'est retournée
        else:
            precision_result = 1
            while precision_result > precision_required:
                current_time = process_time()
                if current_time - t1_start > max_time:
                    ws.range(f"C{col_bissection}").value = "Le temps maximum est dépassé"
                    ws.range(f"D{col_bissection}").value = current_time - t1_start
                    return
                countr = countr + 1
                if countr > ite:
                    break
                x3 = (x1 + x2)/2 #détermine x3 en trouvant le point entre x1 et x2
                x3_context = {"x": x3}
                x3_result = eval(func, globals(), x3_context) #calculer x3 dans la fonction
                bissection_approxs[x3] = x3_result
                if x3_result > 0: #si x3 > 0, alors il devient le nouveau x2, sinon, il devient le nouveau x1
                    x2 = x3
                else:
                    x1 = x3
                bissection_list.append(x3)
                bissection_result = x3
                precision_result = abs(x2-x1)

    except OverflowError:
        bissection_result = "Erreur, résultat des bornes trop élevé"
    except ZeroDivisionError:
        bissection_result = "Processus implique une division par 0"

    t1_stop = process_time() 
    ws.range(f"C{col_bissection}").value = bissection_result #insertion dans le doc excel
    ws.range(f"D{col_bissection}").value = t1_stop - t1_start

    populate_graph_data(inputs, "bissection", bissection_approxs, bissection_result)
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(bissection_approxs, inputs, func,'bissection')
    print('Bissection done')

def secante(inputs: dict, ws: xw.Sheet, min, max):
    max_time = inputs['tempslimite'][0]
    t1_start = process_time() 
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
    ite = iterations    # nbr max d'itérations
    countr = 0  # initialisation du compteur

    if x1_result * x2_result > 0:   # erreur si sécante ne croise pas axe de x
        secante_result = "Aucun zero sur cette section"

    precision_result=1

    while abs(precision_result) > abs(precision_required): # condition de boucle (résultat de f(x3)>0.0001) precision choisie.
        current_time = process_time()
        if current_time - t1_start > max_time:
            ws.range(f"C{col_secante}").value = "Le temps maximum est dépassé"
            ws.range(f"D{col_secante}").value = current_time - t1_start
            return
        countr = countr + 1
        if countr > ite:
            break
        fx1 = eval(func, globals(), {"x": x1})
        fx2 = eval(func, globals(), {"x": x2})
        if fx2 - fx1 ==0:
            secante_result = "Erreur : Division par Zero"
            break
        x3 = x2-(fx2/((fx2-fx1)/(x2-x1)))  # formule secante pour trouver point ou secante touche axe x
        x3_context = {"x": x3}
        x3_result = eval(func, globals(), x3_context)   # valeur trouve pour fx3
        approxs[x3] = x3_result
        secante_list.append(x3)
        precision_result = abs(x3_result) #test pour savoir si on continue boucle ou pas
        x1, x2 = x2, x3 # associe nouvelles valeurs a x1 et x2 pour continuer boucle
        secante_result = x3

    if secante_result > x2 or secante_result < x1:
        secante_result = "La méthode diverge"
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(approxs, inputs, func, 'secante')
    
    ws.range(f"C{col_secante}").value = secante_result
    t1_stop = process_time() 
    ws.range(f"D{col_secante}").value = t1_stop - t1_start
    populate_graph_data(inputs, "secante", approxs, secante_result)
    print('secante done')

### Méthode de Newton ###
def newton(inputs: dict, ws: xw.Sheet, min, max):
     # Initialisation des variables nécessaires à l'algorithmme
    max_time = inputs['tempslimite'][0]
    t1_start = process_time() 
    approxs = {}
    col_newton = inputs['newton'][2]
    func = inputs['fonction'][0]
    precision_required = inputs['precision'][0]

    x1 = inputs['min'][0]
    x1_context = {"x": x1}
    x1_result = eval(func, globals(), x1_context)
    approxs[x1] = x1_result

    min = inputs['min'][0]
    max = inputs['max'][0]

    precision = 1        # On débute avec une précision de 1, elle changera au courant de la boucle
    newton_list = []     # Liste qui stockera les approximations
    x = sp.Symbol("x")

    ite = iterations    # Nombre d'itérations maximal qu'on veut effectuer
    countr = 0          # Compteur d'itérations

    # Boucle qui effectue l'agorithme de Newton
    while precision > precision_required:
        current_time = process_time()
        if current_time - t1_start > max_time:
            ws.range(f"C{col_newton}").value = "Le temps maximum est dépassé"
            ws.range(f"D{col_newton}").value = current_time - t1_start
            return
        countr = countr + 1
        if countr > ite:                            # On sort de la boucle si on dépasse le nombre d'itérations établit
            break
        fx1 = eval(func, globals(), {"x": x1})      # Évaluation de la fonction en x1 (f(x1))
        approxs[x1] = fx1
        deriv = sp.diff(func, x)                    # Calcul de la dérivée en x1 (f'(x1))
        deriv_value = deriv.subs(x, x1)
        x2 = x1 - (fx1 / deriv_value)               # On calcule la prochaine approximation
        fx2 = eval(func, globals(), {"x": x1})
        x1 = x2

        precision = abs(fx2)
        newton_list.append(x2)                      # On ajoute le nouveau point à notre liste
        newton_result = x2
    if newton_result > max*50 or newton_result < min*50:  # La méthode diverge si l'itération sort du domaine d'exploration
        newton_result = "La méthode diverge"

    # Insertion des résultats dans le chiffrier
    ws.range(f"C{col_newton}").value = newton_result
    t1_stop = process_time() 
    ws.range(f"D{col_newton}").value = t1_stop - t1_start
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(approxs, inputs, func, 'newton')
    populate_graph_data(inputs, "newton", approxs, newton_result)

def quasi_newton(inputs: dict, ws: xw.Sheet, min, max):
    #Initialisation des parametres
    max_time = inputs['tempslimite'][0]
    t1_start = process_time() 
    qnewton_approxs = {}
    col_qnewton = inputs['quasinewton'][2]
    func = inputs['fonction'][0]
    precision_required = inputs['precision'][0]
    
    #Conversion de la fonction
    x = sp.Symbol("x")
    func2 = sp.sympify(func)
    x1 = inputs['min'][0]
    x2 = inputs['max'][0]

    f = sp.lambdify(x, func2, 'math')

    precision_result = float('inf')
    qnewton_list = []
    max_iterations = iterations
    iteration = 0

    #Verification de la presence d'une racine
    if f(x1) * f(x2) > 0:
        qnewton_result = "Il n'y a pas de racine dans l'intervalle donné de cette fonction" #pas certain qu'on doit faire ça ?

    #Boucle Quasi-Newton
    while precision_result > precision_required and iteration < max_iterations:
        current_time = process_time()
        if current_time - t1_start > max_time:
            ws.range(f"C{col_qnewton}").value = "Le temps maximum est dépassé"
            ws.range(f"D{col_qnewton}").value = current_time - t1_start
            return
        fx1, fx2 = f(x1), f(x2)

        if fx2 - fx1 == 0:
            return "Erreur : Division par zéro, impossible d'effectuer l'approximation avec la méthode de la sécante."

        x_new = x2 - fx2 * ((x2 - x1) / (fx2 - fx1))
        precision_result = abs(x_new - x2)
        qnewton_list.append(x_new)
        qnewton_approxs[x_new] = f(x_new)

        x1, x2 = x2, x_new
        iteration += 1

    #Verification de la convergence
    qnewton_result = x2 if precision_result <= precision_required else "Aucune convergence"

    #Ecriture des resultats dans excel
    t1_stop = process_time() 
    ws.range(f"D{col_qnewton}").value = t1_stop - t1_start
    ws.range(f"C{col_qnewton}").value = qnewton_result
    
    #Generation des graphiques
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(qnewton_approxs, inputs, func, 'quasi-newton')
    populate_graph_data(inputs, "quasi-newton", qnewton_approxs, qnewton_result)
    print('quasi-newton done')

### Méthode de Muller ###
def muller(inputs: dict, ws: xw.Sheet, min, max):

    try:
        # Initialisation des variables nécessaires à l'algorithmme
        max_time = inputs['tempslimite'][0]
        t1_start = process_time()
        muller_approxs = {}
        col_muller = inputs['muller'][2]
        func = inputs['fonction'][0]
        precision_required = inputs['precision'][0]
        max_iterations = iterations

        # Fonction qui permet d'évaluer la fonction en x
        def f(x):
            context = {"x": x}
            return eval(func, globals(), context)

        # Initialisation du compteur d'itérations et du domaine d'exploration en x
        precision_result = float('inf')
        iteration = 0
        x0 = inputs['min'][0]
        x1 = inputs['max'][0]
        x2 = ((x1-x0)/2)+0.1

        # Boucle de l'algorithme
        while iteration < max_iterations: # On arrête la boucle si on atteint le nombre maximal d'itérations
            current_time = process_time()
            if current_time - t1_start > max_time:
                ws.range(f"C{col_muller}").value = "Le temps maximum est dépassé"
                ws.range(f"D{col_muller}").value = current_time - t1_start
                return
            f0, f1, f2 = f(x0), f(x1), f(x2) # Calcul de la fonctions aux trois points donnés
            muller_approxs[x0] = f0
            muller_approxs[x1] = f1
            muller_approxs[x2] = f2

            # Calcul des coefficients d'interpolation
            h1, h2 = x1 - x0, x2 - x1
            d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
            a = (d2 - d1) / (h2 + h1)
            b = a * h2 + d2
            c = f2

            discriminant = sp.sqrt(b ** 2 - 4 * a * c) # Calcul du discriminant
            denom1, denom2 = b + discriminant, b - discriminant

            # Permet de sélectionner la meilleure valeur de dénominateur
            if abs(denom1) > abs(denom2):
                x3 = x2 - (2 * c) / denom1
            else:
                x3 = x2 - (2 * c) / denom2

            # On évalue la fonciton en x3 ce qui nous donne notre prochaine approximaiton
            print(x3)
            muller_approxs[x3] = float(f(x3))
            print(f(x3))

            # Test qui permet de sortir de la boucle si on atteint la précision escomptée
            if abs(x3 - x2) < precision_required:
                muller_result = x3         # Le résultat est celui de la dernière approximation
                break

            # On ajuste les points en fonctions des nouveaux que nous avons calculés
            x0, x1, x2 = x1, x2, x3
            iteration += 1              # On incrémente le nombre d'itérations

        else:
            muller_result = "No convergence after max iterations" # Si la fonction ne converge pas

    except TypeError:
        muller_result = "La méthode atteint des nombres complexes"

    # Insertion des résultats dans le chiffirer
    ws.range(f"C{col_muller}").value = muller_result
    t1_stop = process_time()
    ws.range(f"D{col_muller}").value = t1_stop - t1_start
    populate_graph_data(inputs, "muller", muller_approxs, muller_result)
    if inputs.get('animationordinateur', [0])[0] == 1:
        add_animated_graph(muller_approxs, inputs, func, 'muller')

def pointfixe(inputs: dict, ws: xw.Sheet, min, max):
    #set up de la fonction et initialisation des paramètres
    max_time = inputs['tempslimite'][0]
    t1_start = process_time() 
    pointfixe_approxs = {}
    col_pointfixe = inputs['pointfixe'][2]
    func = inputs['fonction'][0]
    precision_required = inputs['precision'][0]
    ite = iterations
    x = sp.Symbol("x")
    y = sp.Symbol('y')
    func1 = sp.sympify(func)
    if func1.count(x) > 1 :
        func2_str = str(func1)
        func2_str = func2_str.replace("x", "y", 1)
        func2 = sp.sympify(func2_str)
        gofx = sp.solve(func2, y)  # création de la fonction g(x)
    else:
        gofx = func1

    if type(gofx) == list:
        gofx = gofx[0]

    if gofx.is_real == None:
        try:
            x1 = inputs['min'][0]
            x2 = inputs['max'][0]
            funcprime = sp.diff(gofx,x) # On fait la dérivée de la fonction
            min_funcprime = funcprime.subs(x,x1)# On évalue la dérivée au point min et au point max
            max_funcprime = funcprime.subs(x,x2)
            if min_funcprime.is_real is False: #vérifie que les dérivées donnent des nombres réels
                min_funcprime = 2
            if max_funcprime.is_real is False:
                max_funcprime = 2
            if abs(float(min_funcprime)) < 1 or abs(float(max_funcprime)) < 1: #on vérifie si l'un des deux points possède une dérivée plus petite ou égale à 1
                if float(min_funcprime) <= float(max_funcprime):
                    x0 = x1
                else:
                    x0 = x2
                bissectrice = x #on crée la fonction bissectrice
                print(func1)
                for i in range(1,int(ite)):
                    print(x0)
                    current_time = process_time()
                    if current_time - t1_start > max_time:
                        ws.range(f"C{col_pointfixe}").value = "Le temps maximum est dépassé"
                        ws.range(f"D{col_pointfixe}").value = current_time - t1_start
                        return
                    pointfixe_approxs[x0] = Abs(func1.subs(x,x0).evalf())
                    print(pointfixe_approxs[x0])
                    temp = gofx.subs(x,x0) # on trouve la valeur de x0 dans la fonction
                    x0 = Abs(bissectrice.subs(x,temp).evalf()) # on insert la valeur du y trouvé dans la fonction bissectrice
                pointfixe_result = x0
                print(pointfixe_result)
            else:
                pointfixe_result = gofx
            ws.range(f"C{col_pointfixe}").value = pointfixe_result
            if inputs['animationordinateur'][0] == 1:
                add_animated_graph(pointfixe_approxs, inputs, func, 'pointfixe')
            populate_graph_data(inputs, "pointfixe", pointfixe_approxs, pointfixe_result)
        except:
            pointfixe_result = "Echec, n'a pas trouvé de racine"
            ws.range(f"C{col_pointfixe}").value = pointfixe_result
    else:
        pointfixe_approxs[0] = func1.subs(x, gofx)
        pointfixe_result = gofx
        ws.range(f"C{col_pointfixe}").value = pointfixe_result
    t1_stop = process_time() 
    ws.range(f"D{col_pointfixe}").value = t1_stop - t1_start
    print("pointfixe done")

def find_root():
    for value in root_plot_data.values():
        if isinstance(value, (int, float)):  # Check if the value is a number (int or float)
            return value
    return None  # If no valid number is found


def initiate_plot(nb_of_plot: int):
    if(nb_of_plot == 1):
        return plt.subplots(figsize=(6, 4))
    return plt.subplots(1, nb_of_plot, figsize=(12, 4))

def add_root_plot(axs: matplotlib.axes, nb_of_plot, fonction, min, max):
    plot = axs if nb_of_plot == 1 else axs[0]

    x = np.linspace(int(min), int(max), 1000) #graph the function through 1000 points between max and min
    y = [eval(fonction[0], globals(), {'x':i}) for i in x]
    root_y = 0
    root_x = find_root()
    plot.scatter(root_x, root_y, color='red', zorder=5)  # zorder ensures it's on top
# Annotate the point

    plot = axs if nb_of_plot == 1 else axs[0]
    plot.plot(x, y, color='blue')
    plot.set_title('Racine de fonction')
    plot.set_xlabel('x')
    plot.set_ylabel('y')
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


def add_approx_sheet(ws: xw.Sheet):
    ws.range('A1').expand().clear()
    print(approxs_plot_data)
    row = 1
    col = 1
    cell = 'B'
    i = 0
    for key, values in approxs_plot_data.items():
        ws.range(f"{chr(ord(cell) + i)}{col}").value = key
        z = 1
        ws.range(f"{chr(ord(cell) + i)}{col + z}").value = 'x'
        ws.range(f"{chr(ord(cell) + i + 1)}{col + z}").value = 'y'
        z = z+1
        iter = 1
        for x, y in values.items():
            ws.range(f"A{col+z}").value = f"Iteration {iter}"
            ws.range(f"{chr(ord(cell) + i)}{col + z}").value = x
            ws.range(f"{chr(ord(cell) + i + 1)}{col + z}").value = y
            z = z + 1
            iter = iter+1
        i = i + 3



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

    try:
        wb.sheets["Approximations"].delete()
    except Exception:
        print("Approximations sheet doesn't exist")

    ws_approx = wb.sheets.add(name="Approximations", after=output)

    
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
            add_root_plot(axes, nb_of_plot, input_data['fonction'], x_min[0], x_max[0])
        if input_data['graphiqueapproximations'][0] == 1:
            add_approx_plot(axes, nb_of_plot)
        ws.pictures.add(fig, name='Graphiques', update=True, left=ws.range('E8').left, top=ws.range('E8').top)

    #add the animated graph je vais creer une loop pour le faire pour chacun des graphs plus tard.
    if input_data['animationordinateur'][0] == 1:
        gif_list = []
        y=8
        if input_data['bissection'][0] == 1: gif_list.append(f'bissection_animation')
        if input_data['secante'][0] == 1: gif_list.append(f'secante_animation')
        if input_data['newton'][0] == 1: gif_list.append(f'newton_animation')
        if input_data['quasinewton'][0] == 1: gif_list.append(f'quasi-newton_animation')
        if input_data['muller'][0] == 1: gif_list.append(f'muller_animation')
        if input_data['pointfixe'][0]: gif_list.append(f'pointfixe_animation')

        for i in gif_list:
            try:
                script_dir = os.path.dirname(__file__)  # folder where this .py file lives
                gif_path = os.path.join(script_dir, f"{i}.gif")
                ws.pictures.add(gif_path, left=ws.range(f"S{y}").left, top= ws.range(f'S{y}').top)
                y += 9
            except:
                print(f"{i} did not work")

    ws_approx.range('A1').expand().clear()
    add_approx_sheet(ws_approx)
    print(f"Les données ont été ajoutées à l'onglet '{output}' du fichier '{file_name}'.")


# pip install -r requirements.txt -> pour les dependencies
# Si vous voulez run ex: python .\root_functions.py .\Devoir1_Entame.xlsx
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\nunix: python root_functions.py <csv_file_name>\nwin: python .\\root_functions.py .\\<csv_file_name>")
        sys.exit(1)
    handle_inputs(sys.argv[1])
