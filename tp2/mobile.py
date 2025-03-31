#!/usr/bin/env python

#Imports
import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# input pour le systeme d'équation (à mettre à jour avec le ??Excel??)
a11 = 1
a12 = -3
b1 = 0
a21 = 0.25
a22 = 3
b2 = 0
y11= 1 #y1 au temps t
y22 = 2 #y2 au temps t
t = 1 #point a temps t
h_min = -1 #permet de changer l'horizon min
h_max = 1 #pemet de changer l'horizon max
laps = 0.01 #permet de changer l'espace entre les estimations des graphs
fps = 40 #permet de changer la vitesse des graphs


#créer les system d'équation
def sys_equation(a11,a12,b1,a21,a22,b2):
    y1, y2 = sp.symbols('y1 y2')
    y1_prime = a11 * y1 + a12 * y2 + b1
    y2_prime = a21 * y1 + a22 * y2 + b2
    return  y1_prime , y2_prime

#Créer la solution particulière
def x_equation(a11,a12,b1,a21,a22,b2):
    x1, x2 = sp.symbols('x1 x2')
    x1 = (-b1*a22 + b2*a12)/(a11*a22 - a12*a21)
    x2 = (-b2*a11 + b1*a21)/(a11*a22 - a12*a21)
    return x1, x2

#Créer la solution homogène
def h1_h2_primeprime(a11, a12, a21, a22):
    t = sp.symbols('t')
    h1 = sp.Function('h1')(t)
    h2 = sp.Function('h2')(t)
    h1p = sp.Function('h1p')(t)
    h2p = sp.Function('h2p')(t)
    h1pp = sp.Function('h1pp')(t)
    h2pp = sp.Function('h2pp')(t)
    h1pp = (a11 + a22) * h1p + (a12*a21 - a11*a22)*h1
    a1 = a11 + a22
    a2 = a12 * a21 - a11 * a22
    eq_h1 = sp.Eq(
        h1.diff(t, 2),
        (a1) * h1.diff(t) + (a2) * h1
    )
    sol_h1 = sp.dsolve(eq_h1).rhs
    h1 = sol_h1
    sol_h1p = sp.diff(sol_h1, t)
    h2 = (sol_h1p - a11 * sol_h1) / a12
    return h1, h2

#Merge la solution particulière et la solution homogène
def merge_ht_xt(h1,h2,x1,x2):
    y1 = h1 + x1
    y2 = h2 + x2
    return y1, y2


#Résoudre les constantes
def solve_constant(y1ct,y2ct,y11,y22,t_val):
    C1, C2 = sp.symbols('C1 C2')
    t = sp.symbols('t')
    y1c = y1ct.subs(t, t_val)
    y2c = y2ct.subs(t, t_val)

    eq1 = sp.Eq(y1c,y11)
    eq2 = sp.Eq(y2c,y22)
    sol = sp.solve((eq1,eq2),(C1,C2))

    c1 = sol[C1]
    c2 = sol[C2]

    y1t = y1ct.subs(C1, c1)
    y1t = y1t.subs(C2, c2)
    y2t = y2ct.subs(C1, c1)
    y2t = y2t.subs(C2, c2)

    return c1, c2, y1c, y2c, y1t, y2t, y1ct, y2ct

#Fonction inutilisée en ce moment
def build_graph(y1t, y2t, h_min, h_max, laps):
    t = sp.Symbol('t')
    t_vals = []
    f1_vals = []
    f2_vals = []
    for x in np.arange(h_min, h_max + laps, laps):
        t_vals.append(x)
        f1_vals.append(y1t.subs(t, x).evalf())
        f2_vals.append(y2t.subs(t, x).evalf())

    # Plot 1: f1_vals
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, f1_vals, label="f1(t)", color='blue')
    plt.title("Graph of f1(t)")
    plt.xlabel("t")
    plt.ylabel("f1(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: f2_vals
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, f2_vals, label="f2(t)", color='green')
    plt.title("Graph of f2(t)")
    plt.xlabel("t")
    plt.ylabel("f2(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 3: f1(t) + f2(t)
    plt.figure(figsize=(10, 6))
    plt.plot(f1_vals, f2_vals, label="f2(t)", color='purple')
    plt.title("Graph of f1(t) + f2(t)")
    plt.xlabel("f1(t)")
    plt.ylabel("f1(t) and f2(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 4: f1_vals and f2_vals on same graph
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, f1_vals, label="f1(t)", color='blue')
    plt.plot(t_vals, f2_vals, label="f2(t)", color='green')
    plt.title("f1(t) and f2(t) on the same graph")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

def build_animated_graph(y1t, y2t, h_min, h_max, laps, fps, name_prefix="graph"):
    t = sp.Symbol('t')
    t_vals = []
    f1_vals = []
    f2_vals = []

    for x in np.arange(h_min, h_max + laps, laps):
        t_vals.append(x)
        f1_vals.append(float(y1t.subs(t, x).evalf()))
        f2_vals.append(float(y2t.subs(t, x).evalf()))

    # Output path
    script_dir = os.path.dirname(__file__)

    # Helper function to animate and save each plot
    def animate_and_save(x_vals, y_vals, xlabel, ylabel, title, filename, second_y_vals=None):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], color='blue', label=title)
        ax.set_xlim(min(x_vals), max(x_vals))
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        def update(frame):
            line.set_data(x_vals[:frame + 1], y_vals[:frame + 1])
            return line,

        anim = FuncAnimation(fig, update, frames=len(x_vals), interval=1, blit=True)  # reduced frames
        gif_path = os.path.join(script_dir, f"{name_prefix}_{filename}.gif")
        anim.save(gif_path, writer='pillow', fps=fps)  # increased fps
        plt.close()

    # Animate Plot 1: f1(t)
    animate_and_save(t_vals, f1_vals, "t", "f1(t)", "Graph of f1(t)", "f1")

    # Animate Plot 2: f2(t)
    animate_and_save(t_vals, f2_vals, "t", "f2(t)", "Graph of f2(t)", "f2")

    # Animate Plot 3: f1(t) vs f2(t)
    animate_and_save(f1_vals, f2_vals, "f1(t)", "f2(t)", "Graph of f1(t) vs f2(t)", "f1_vs_f2")

    # Animate Plot 4: f1(t) and f2(t) on same graph
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], color='blue', label='f1(t)')
    line2, = ax.plot([], [], color='green', label='f2(t)')
    ax.set_xlim(min(t_vals), max(t_vals))
    ax.set_ylim(min(min(f1_vals), min(f2_vals)) - 1, max(max(f1_vals), max(f2_vals)) + 1)
    ax.set_title("f1(t) and f2(t) on the same graph")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    def update_combined(frame):
        line1.set_data(t_vals[:frame + 1], f1_vals[:frame + 1])
        line2.set_data(t_vals[:frame + 1], f2_vals[:frame + 1])
        return line1, line2

    anim = FuncAnimation(fig, update_combined, frames=len(t_vals), interval=1, blit=True)  # reduced frames
    gif_path = os.path.join(script_dir, f"{name_prefix}_combined.gif")
    anim.save(gif_path, writer='pillow', fps=fps)  # increased fps
    plt.close()

def run_model(a11,a12,b1,a21,a22,b2,y11,y22,t, h_min, h_max, laps, fps):
    y1_prime , y2_prime = sys_equation(a11,a12,b1,a21,a22,b2)
    x1, x2 = x_equation(a11,a12,b1,a21,a22,b2)
    h1, h2 = h1_h2_primeprime(a11,a12,a21,a22)
    y1t, y2t = merge_ht_xt(h1,h2,x1,x2)
    c1, c2, y1c, y2c, y1t, y2t, y1ct, y2ct = solve_constant(y1t,y2t,y11,y22,t)
    build_animated_graph(y1t, y2t, h_min, h_max, laps, fps)

if __name__ == "__main__":
    # Execution
    run_model(a11, a12, b1, a21, a22, b2, y11, y22, t, h_min, h_max, laps, fps)