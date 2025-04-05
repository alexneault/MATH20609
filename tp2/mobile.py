#!/usr/bin/env python

# Imports
import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import messagebox
from tkinter import Entry, Label
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math_functions import *

root = tk.Tk()
inputs_entries: list[tuple[Entry, Label]] = []
inputs = {"a11": 1, "a12": -3, "b1": 0, "a21": 0.25, "a22": 3, "b2": 0, "y1,t": 1, "y2,t": 2, "t": 1, "h_min": -1, "h_max": 1, "laps": 0.01, "fps": 40, "intervals": 40}  # default interval = fps

anim = None #declare anim as global variable.
output_labels = []
canvas_widget = None
current_animation = None


def build_animated_graph(y1t, y2t, h_min, h_max, laps, fps, canvas_widget):
    t = sp.Symbol('t')
    t_vals = []
    f1_vals = []
    f2_vals = []

    for x in np.arange(h_min, h_max + laps, laps):
        t_vals.append(x)
        f1_vals.append(float(y1t.subs(t, x).evalf()))
        f2_vals.append(float(y2t.subs(t, x).evalf()))

    fig, ax = plt.subplots()
    line1, = ax.plot([], [], color='blue', label='y1(t)')
    line2, = ax.plot([], [], color='green', label='y2(t)')
    ax.set_xlim(min(t_vals), max(t_vals))
    ax.set_ylim(min(min(f1_vals), min(f2_vals)) - 1, max(max(f1_vals), max(f2_vals)) + 1)
    ax.set_title("Comportement de y1(t) et y2(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("Valeur")
    ax.grid(True)
    ax.legend()

    def update_combined(frame):
        line1.set_data(t_vals[:frame + 1], f1_vals[:frame + 1])
        line2.set_data(t_vals[:frame + 1], f2_vals[:frame + 1])
        return line1, line2

    global current_animation
    current_animation = FuncAnimation(fig, update_combined, frames=len(t_vals), interval=inputs["intervals"], blit=True)

    canvas = FigureCanvasTkAgg(fig, master=canvas_widget) #use canvas_widget
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Interval', 1, 100, valinit=inputs["intervals"], valstep=1)
    slider.on_changed(lambda val: update_interval(val, slider, fig, update_combined, t_vals))
    return canvas

def build_animated_graph2(y1t, y2t, h_min, h_max, laps, fps, canvas_widget):
    t = sp.Symbol('t')
    t_vals = []
    f1_vals = []
    f2_vals = []

    for x in np.arange(h_min, h_max + laps, laps):
        t_vals.append(x)
        f1_vals.append(float(y1t.subs(t, x).evalf()))
        f2_vals.append(float(y2t.subs(t, x).evalf()))

    fig, ax = plt.subplots()
    line1, = ax.plot([], [], color='purple', label='y1(t) en fonction de y2(t)') # Changed color and label
    ax.set_xlim(min(f1_vals) + 0.2 * min(f1_vals), max(f1_vals) + 0.2 * max(f1_vals)) # Set x-axis to f1 values
    ax.set_ylim(min(f2_vals) + 0.2 * min(f2_vals), max(f2_vals) + 0.2 * max(f2_vals)) # Set y-axis to f2 values
    ax.set_title("y1t et y2t") # Changed title
    ax.set_xlabel("y1(t)") # Changed x-axis label
    ax.set_ylabel("y2(t)") # Changed y-axis label
    ax.grid(True)
    ax.legend()

    def update_combined(frame):
        line1.set_data(f1_vals[:frame + 1], f2_vals[:frame + 1])
        return line1,

    global current_animation
    current_animation = FuncAnimation(fig, update_combined, frames=len(t_vals), interval=inputs["intervals"], blit=True)

    canvas = FigureCanvasTkAgg(fig, master=canvas_widget) #use canvas_widget
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Interval', 1, 100, valinit=inputs["intervals"], valstep=1)
    slider.on_changed(lambda val: update_interval(val, slider, fig, update_combined, f1_vals)) # Changed t_vals to f1_vals
    return canvas




def update_interval(val, slider, fig, update_combined, t_vals):
    new_interval = slider.val
    global current_animation
    if current_animation:
        current_animation.event_source.stop()
    inputs["intervals"] = new_interval
    current_animation = FuncAnimation(fig, update_combined, frames=len(t_vals), interval=inputs["intervals"], blit=True)
    current_animation.event_source.start()
    fig.canvas.draw()

def clear_previous_results():
    global output_labels
    for label in output_labels:
        label.destroy()
    output_labels = []
    global current_animation
    if current_animation:
        current_animation.event_source.stop()
        plt.close(current_animation._fig)
        current_animation = None
    global canvas_widget
    if canvas_widget:
        for widget in canvas_widget.winfo_children():
            widget.destroy()

def run_model(a11, a12, b1, a21, a22, b2, y11, y22, t, h_min, h_max, laps, fps, canvas_widget):
    y1_prime, y2_prime = sys_equation(a11, a12, b1, a21, a22, b2)
    x1, x2 = x_equation(a11, a12, b1, a21, a22, b2)
    if x1 is None or x2 is None:
        output_label = tk.Label(root, text="Le système n'a pas de point fixe")
        output_labels.append(output_label)
        output_label.pack()
        return

    h1, h2 = h1_h2_primeprime(a11, a12, a21, a22)
    y1t_unsolved, y2t_unsolved = merge_ht_xt(h1, h2, x1, x2)
    y1t_unsolved_r = y1t_unsolved.evalf(4)
    y2t_unsolved_r = y2t_unsolved.evalf(4)

    if y1t_unsolved is None or y2t_unsolved is None:
        output_label = tk.Label(root, text="Les solutions particulière et homogène n'ont pas pu être résolue.")
        output_labels.append(output_label)
        output_label.pack()
        return

    output_label1 = tk.Label(root, text=f"y1(t) general solution: {y1t_unsolved_r}")
    output_labels.append(output_label1)
    output_label1.pack()
    output_label2 = tk.Label(root, text=f"y2(t) general solution: {y2t_unsolved_r}")
    output_labels.append(output_label2)
    output_label2.pack()

    if not isinstance(y11, (int, float)) or not isinstance(y22, (int, float)) or not isinstance(t, (int, float)):
        output_label3 = tk.Label(root, text="Cannot generate specific solution without initial conditions (y11, y22, t).")
        output_labels.append(output_label3)
        output_label3.pack()
    else:
        c1, c2, y1c, y2c, y1t_solved, y2t_solved, y1ct, y2ct = solve_constant(y1t_unsolved, y2t_unsolved, y11, y22, t)
        if c1 is not None and c2 is not None:
            y1t_solved_r = y1t_solved.evalf(4)
            y2t_solved_r = y2t_solved.evalf(4)

            output_label4 = tk.Label(root, text=f"y1(t) specific solution: {y1t_solved_r}")
            output_labels.append(output_label4)
            output_label4.pack()
            output_label5 = tk.Label(root, text=f"y2(t) specific solution: {y2t_solved_r}")
            output_labels.append(output_label5)
            output_label5.pack()
            build_animated_graph(y1t_solved, y2t_solved, h_min, h_max, laps, fps, canvas_widget)
        else:
            output_label_const_err = tk.Label(root, text="Could not solve for constants C1 and C2.")
            output_labels.append(output_label_const_err)
            output_label_const_err.pack()

def run_model_for_graph2(a11, a12, b1, a21, a22, b2, y11, y22, t, h_min, h_max, laps, fps, canvas_widget):
    y1_prime, y2_prime = sys_equation(a11, a12, b1, a21, a22, b2)
    x1, x2 = x_equation(a11, a12, b1, a21, a22, b2)
    if x1 is None or x2 is None:
        output_label = tk.Label(root, text="System has no unique fixed point (determinant is zero).")
        output_labels.append(output_label)
        output_label.pack()
        return

    h1, h2 = h1_h2_primeprime(a11, a12, a21, a22)
    y1t_unsolved, y2t_unsolved = merge_ht_xt(h1, h2, x1, x2)

    if y1t_unsolved is None or y2t_unsolved is None:
        output_label = tk.Label(root, text="Could not merge homogeneous and particular solutions.")
        output_labels.append(output_label)
        output_label.pack()
        return
    y1t_unsolved_r = y1t_unsolved.evalf(4)
    y2t_unsolved_r = y2t_unsolved.evalf(4)

    output_label1 = tk.Label(root, text=f"y1(t) general solution: {y1t_unsolved_r}")
    output_labels.append(output_label1)
    output_label1.pack()
    output_label2 = tk.Label(root, text=f"y2(t) general solution: {y2t_unsolved_r}")
    output_labels.append(output_label2)
    output_label2.pack()

    if not isinstance(y11, (int, float)) or not isinstance(y22, (int, float)) or not isinstance(t, (int, float)):
        output_label3 = tk.Label(root, text="Cannot generate specific solution without initial conditions (y11, y22, t).")
        output_labels.append(output_label3)
        output_label3.pack()
    else:
        c1, c2, y1c, y2c, y1t_solved, y2t_solved, y1ct, y2ct = solve_constant(y1t_unsolved, y2t_unsolved, y11, y22, t)
        if c1 is not None and c2 is not None:
            y1t_solved_r = y1t_solved.evalf(4)
            y2t_solved_r = y2t_solved.evalf(4)
            output_label4 = tk.Label(root, text=f"y1(t) specific solution: {y1t_solved_r}")
            output_labels.append(output_label4)
            output_label4.pack()
            output_label5 = tk.Label(root, text=f"y2(t) specific solution: {y2t_solved_r}")
            output_labels.append(output_label5)
            output_label5.pack()
            # Call the second graph building function here
            build_animated_graph2(y1t_solved, y2t_solved, h_min, h_max, laps, fps, canvas_widget)
        else:
            output_label_const_err = tk.Label(root, text="Could not solve for constants C1 and C2.")
            output_labels.append(output_label_const_err)
            output_label_const_err.pack()


def generate_model(canvas_widget_param):
    clear_previous_results()
    global canvas_widget
    canvas_widget = canvas_widget_param
    try:
        for (entry, label) in inputs_entries:
            name = label['text']
            text_value = entry.get().strip()
            if name in ["y1,t", "y2,t", "t"] and text_value == "":
                inputs[name] = None
            else:
                try:
                    inputs[name] = float(text_value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid input for '{name}'. Please enter a number for '{name}'.")
                    return
        run_model(inputs["a11"], inputs["a12"], inputs["b1"], inputs["a21"], inputs["a22"], inputs["b2"], inputs["y1,t"], inputs["y2,t"], inputs["t"], inputs["h_min"], inputs["h_max"], inputs["laps"], inputs["fps"], canvas_widget)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers in all fields.")


def generate_model2(canvas_widget_param):
    clear_previous_results()
    global canvas_widget
    canvas_widget = canvas_widget_param
    try:
        for (entry, label) in inputs_entries:
            name = label['text']
            text_value = entry.get().strip()
            if name in ["y11", "y22", "t"] and text_value == "":
                inputs[name] = None
            else:
                try:
                    inputs[name] = float(text_value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid input for '{name}'. Please enter a number for '{name}'.")
                    return
        run_model_for_graph2(inputs["a11"], inputs["a12"], inputs["b1"], inputs["a21"], inputs["a22"], inputs["b2"], inputs["y11"], inputs["y22"], inputs["t"], inputs["h_min"], inputs["h_max"], inputs["laps"], inputs["fps"], canvas_widget)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers in all fields.")

def display_equation_image(parent, filename):
    try:
        img = tk.PhotoImage(file=filename)
        label = tk.Label(parent, image=img)
        label.image = img # Keep a reference!
        label.pack(side=tk.BOTTOM, padx=10, pady=10, anchor='sw') # Pack to the bottom, anchor bottom-left
    except tk.TclError:
        error_label = tk.Label(parent, text=f"Error loading {filename}")
        error_label.pack(side=tk.BOTTOM, padx=10, pady=10, anchor='sw') # Pack to the bottom, anchor bottom-left

def run_tkinter():
    root.title("Mobile generation")
    root.geometry("1800x900")  # Increased window size

    equation_image_filename = 'Sys_equation.PNG'

    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, anchor='nw') # Inputs on the left

    global canvas_widget
    canvas_widget = tk.Frame(root)
    canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1, padx=10, pady=10, anchor='nw') # Plots on the right
    root.columnconfigure(1, weight=1)

    # Display the equation image below the input frame
    display_equation_image(input_frame, equation_image_filename)

    for key in inputs.keys():
        label = tk.Label(input_frame, text=key)
        label.pack(anchor='nw', pady=2)
        entry = tk.Entry(input_frame)
        entry.insert(0, str(inputs[key]))
        entry.pack(pady=2, anchor='nw', padx=5)
        inputs_entries.append((entry, label))

    submit_btn = tk.Button(input_frame, text="Générer le graphique comportement individuel", command=lambda: generate_model(canvas_widget))
    submit_btn.pack(pady=10, anchor='nw')
    submit_btn2 = tk.Button(input_frame, text="Générer le graphique comportement combiné",
                           command=lambda: generate_model2(canvas_widget))

    submit_btn2.pack(pady=10, anchor='nw')
    root.mainloop()
if __name__ == "__main__":
    run_tkinter()