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

root = tk.Tk()
inputs_entries: list[tuple[Entry, Label]] = []
inputs = {"a11": 1, "a12": -3, "b1": 0, "a21": 0.25, "a22": 3, "b2": 0, "y11": 1, "y22": 2, "t": 1, "h_min": -1, "h_max": 1, "laps": 0.01, "fps": 40, "intervals": 40}  # default interval = fps

anim = None #declare anim as global variable.
output_labels = []
canvas_widget = None
current_animation = None

def sys_equation(a11, a12, b1, a21, a22, b2):
    y1, y2 = sp.symbols('y1 y2')
    y1_prime = a11 * y1 + a12 * y2 + b1
    y2_prime = a21 * y1 + a22 * y2 + b2
    return y1_prime, y2_prime

def x_equation(a11, a12, b1, a21, a22, b2):
    x1, x2 = sp.symbols('x1 x2')
    det_A = a11 * a22 - a12 * a21
    if det_A == 0:
        return None, None  # Handle singular case
    x1 = (-b1 * a22 + b2 * a12) / det_A
    x2 = (-b2 * a11 + b1 * a21) / det_A
    return x1, x2

def h1_h2_primeprime(a11, a12, a21, a22):
    t = sp.symbols('t')
    h1 = sp.Function('h1')(t)
    h2 = sp.Function('h2')(t)
    h1p = sp.Function('h1p')(t)
    h2p = sp.Function('h2p')(t)
    h1pp = sp.Function('h1pp')(t)
    h2pp = sp.Function('h2pp')(t)
    h1pp = (a11 + a22) * h1p + (a12 * a21 - a11 * a22) * h1
    a1 = a11 + a22
    a2 = a12 * a21 - a11 * a22
    eq_h1 = sp.Eq(h1.diff(t, 2), (a1) * h1.diff(t) + (a2) * h1)
    sol_h1 = sp.dsolve(eq_h1).rhs
    h1 = sol_h1
    sol_h1p = sp.diff(sol_h1, t)
    h2 = (sol_h1p - a11 * sol_h1) / a12
    return h1, h2

def merge_ht_xt(h1, h2, x1, x2):
    if x1 is None or x2 is None:
        return None, None
    y1 = h1 + x1
    y2 = h2 + x2
    return y1, y2

def solve_constant(y1ct, y2ct, y11, y22, t_val):
    C1, C2 = sp.symbols('C1 C2')
    t = sp.symbols('t')
    y1c = y1ct.subs(t, t_val)
    y2c = y2ct.subs(t, t_val)
    eq1 = sp.Eq(y1c, y11)
    eq2 = sp.Eq(y2c, y22)
    sol = sp.solve((eq1, eq2), (C1, C2))
    if not sol:
        return None, None, None, None, None, None, None, None
    c1 = sol[C1]
    c2 = sol[C2]
    y1t = y1ct.subs(C1, c1).subs(C2, c2)
    y2t = y2ct.subs(C1, c1).subs(C2, c2)
    return c1, c2, y1c, y2c, y1t, y2t, y1ct, y2ct

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
        line1.set_data(f1_vals[:frame + 1], f2_vals[:frame + 1])
        line2.set_data(f1_vals[:frame + 1], f2_vals[:frame + 1])
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

    output_label1 = tk.Label(root, text=f"y1(t) general solution: {y1t_unsolved}")
    output_labels.append(output_label1)
    output_label1.pack()
    output_label2 = tk.Label(root, text=f"y2(t) general solution: {y2t_unsolved}")
    output_labels.append(output_label2)
    output_label2.pack()

    if not isinstance(y11, (int, float)) or not isinstance(y22, (int, float)) or not isinstance(t, (int, float)):
        output_label3 = tk.Label(root, text="Cannot generate specific solution without initial conditions (y11, y22, t).")
        output_labels.append(output_label3)
        output_label3.pack()
    else:
        c1, c2, y1c, y2c, y1t_solved, y2t_solved, y1ct, y2ct = solve_constant(y1t_unsolved, y2t_unsolved, y11, y22, t)
        if c1 is not None and c2 is not None:
            output_label4 = tk.Label(root, text=f"y1(t) specific solution: {y1t_solved}")
            output_labels.append(output_label4)
            output_label4.pack()
            output_label5 = tk.Label(root, text=f"y2(t) specific solution: {y2t_solved}")
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

    output_label1 = tk.Label(root, text=f"y1(t) general solution: {y1t_unsolved}")
    output_labels.append(output_label1)
    output_label1.pack()
    output_label2 = tk.Label(root, text=f"y2(t) general solution: {y2t_unsolved}")
    output_labels.append(output_label2)
    output_label2.pack()

    if not isinstance(y11, (int, float)) or not isinstance(y22, (int, float)) or not isinstance(t, (int, float)):
        output_label3 = tk.Label(root, text="Cannot generate specific solution without initial conditions (y11, y22, t).")
        output_labels.append(output_label3)
        output_label3.pack()
    else:
        c1, c2, y1c, y2c, y1t_solved, y2t_solved, y1ct, y2ct = solve_constant(y1t_unsolved, y2t_unsolved, y11, y22, t)
        if c1 is not None and c2 is not None:
            output_label4 = tk.Label(root, text=f"y1(t) specific solution: {y1t_solved}")
            output_labels.append(output_label4)
            output_label4.pack()
            output_label5 = tk.Label(root, text=f"y2(t) specific solution: {y2t_solved}")
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
            if name in ["y11", "y22", "t"] and text_value == "":
                inputs[name] = None
            else:
                try:
                    inputs[name] = float(text_value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid input for '{name}'. Please enter a number for '{name}'.")
                    return
        run_model(inputs["a11"], inputs["a12"], inputs["b1"], inputs["a21"], inputs["a22"], inputs["b2"], inputs["y11"], inputs["y22"], inputs["t"], inputs["h_min"], inputs["h_max"], inputs["laps"], inputs["fps"], canvas_widget)
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


def run_tkinter():
    root.title("Mobile generation")
    root.geometry("1500x700")

    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.LEFT, fill=tk.Y)

    global canvas_widget
    canvas_widget = tk.Frame(root)
    canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1, padx=5, pady=5) # Add padding
    root.columnconfigure(1, weight=1) # Configure the second column (index 1) to have weight


    for key in inputs.keys():
        label = tk.Label(input_frame, text=key)
        label.pack()
        entry = tk.Entry(input_frame)
        entry.insert(0, str(inputs[key]))
        entry.pack(pady=2)
        inputs_entries.append((entry, label))

    submit_btn = tk.Button(input_frame, text="Create y1 and y2 combined graph", command=lambda: generate_model(canvas_widget))
    submit_btn.pack(pady=10)
    submit_btn2 = tk.Button(input_frame, text="Create y1 and y2 combined graph v2",
                           command=lambda: generate_model2(canvas_widget))
    submit_btn2.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    run_tkinter()