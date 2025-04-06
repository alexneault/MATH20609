#!/usr/bin/env python

import tkinter as tk
from tkinter import messagebox
from tkinter import Entry, Label

root = tk.Tk()
resource_entry = tk.Entry(root, width=10)
resource_entry.insert(0, "50")
hour_entries = []
max_benefit_label = tk.Label(root, text="Maximum Benefit: ", font=('Arial', 10))
allocation_plan_label = tk.Label(root, text="Resource Allocation Plan: ", font=('Arial', 10))

inputs = {"comptabilite": ["D","C","B","A"], "finance": ["C","B","A","A"], "science_decision": ["D","C","B","A"], "gestion": ["B","B","A","A"]}
entry_widgets = []
grades = {"A" :5, "B":4, "C":3, "D":1}

grades_val_to_grade = { 5 :"A", 4:"B", 3:"C", 1:"D"}

def copy_with_zeros(matrix):
    return [[0] * len(matrix[0]) for _ in range(len(matrix))]

def resource_allocation_dp(resources: int, benefit_matrix: list[list[int]], resources_matrix: list[list[int]]) -> tuple[int, list[int]]:
    tasks = len(benefit_matrix)
    dp = [[-1] * (resources + 1) for _ in range(tasks)]
    allocation = [[0] * (resources + 1) for _ in range(tasks)]

    for r in range(resources + 1):
        for k, cost in enumerate(resources_matrix[0]):
            if cost <= r and cost >= 1:  # Ensure at least 1 block
                dp[0][r] = max(dp[0][r], benefit_matrix[0][k])
                if dp[0][r] == benefit_matrix[0][k]:
                    allocation[0][r] = cost

    for i in range(1, tasks):
        for r in range(resources + 1):
            max_benefit = -1
            best_alloc = 0
            for k, cost in enumerate(resources_matrix[i]):
                if cost <= r and cost >= 1:  # Only allow cost ≥ 1
                    prev_benefit = dp[i - 1][r - cost]
                    if prev_benefit != -1:
                        benefit = benefit_matrix[i][k] + prev_benefit
                        if benefit > max_benefit:
                            max_benefit = benefit
                            best_alloc = cost
            dp[i][r] = max_benefit
            allocation[i][r] = best_alloc

    if dp[tasks - 1][resources] == -1:
        return 0, [0] * tasks  # No valid allocation

    # Backtrack
    r = resources
    alloc_plan = [0] * tasks
    for i in range(tasks - 1, -1, -1):
        alloc_plan[i] = allocation[i][r]
        r -= allocation[i][r]

    return dp[tasks - 1][resources], alloc_plan


# Validation function
def validate_char(char):
    return char in ('A', 'B', 'C', 'D', 'a', 'b', 'c', 'd', '')
def generate_benefice_matrix(data: dict):
    matrix = []
    for val in data.values():
        matrix.append(list(map(grades_to_value, val)))

    return matrix



# Function to handle button click
def on_submit():
    updated_data = {}
    for i, key in enumerate(inputs.keys()):
        updated_row = []
        for entry in entry_widgets[i]:
            try:
                val = str(entry.get())
            except ValueError:
                val = entry.get()  # fallback to string if not a number
            updated_row.append(val.upper())
        updated_data[key] = updated_row

    #print("Updated data:")
    #print(updated_data)
    resource_value = int(resource_entry.get())
    #print("Resource value:", resource_value)

    updated_benefit_matrix = generate_benefice_matrix(updated_data)
    dynamic_resources_matrix = get_resource_matrix_from_ui()
    max_benefit, allocation_plan = resource_allocation_dp(resource_value, updated_benefit_matrix, dynamic_resources_matrix)
    max_benefit_label.config(text=f"Maximum Benefit: {max_benefit}")
    allocation_plan_label.config(text=f"Resource Allocation Plan: {allocation_plan_pretty(allocation_plan, updated_benefit_matrix)}")

def allocation_plan_pretty(plan: list[int], update_benefit_matrix):
    print(plan)
    resource_matrix = get_resource_matrix_from_ui()
    course_names = ["Comptabilité", "Finance", "Science de la décision", "Gestion"]
    
    res = ""
    for i, hours in enumerate(plan):
        try:
            index_in_row = resource_matrix[i].index(hours)
            grade_val = update_benefit_matrix[i][index_in_row]
            grade = grades_val_to_grade[grade_val]
        except ValueError:
            grade = "E"  # No valid allocation
        res += f"\n{course_names[i]}: {grade}"
    
    return res

def get_resource_matrix_from_ui():
    return [list(map(int, [entry.get() for entry in hour_entries])) for _ in range(4)]
    

def run_tkinter():
    default_grades = [
        inputs["comptabilite"],
        inputs["finance"],
        inputs["science_decision"],
        inputs["gestion"], # Cours 1
    ]
    root.title("Allocation generation")
    root.geometry("1800x900")  # Increased window size
    #input_frame = tk.Frame(root)
    #input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, anchor='nw') # Inputs on the left
    # Store entries for later access if needed
    vcmd = (root.register(validate_char), '%P')
    # Display header
    header = ["Cours", "5 Heures", "10 Heures", "15 Heures", "20 Heures"]
    for col, text in enumerate(header):
        label = tk.Label(root, text=text, font=('Arial', 10, 'bold'), borderwidth=1, relief="solid", padx=5, pady=5)
        label.grid(row=0, column=col, sticky="nsew")

    # Display data rows
    for row, (key, row_values) in enumerate(zip(inputs.keys(), default_grades), start=1):
        row_entries = []

        # Display key
        label = tk.Label(root, text=key, borderwidth=1, relief="solid", padx=5, pady=5)
        label.grid(row=row+1, column=0, sticky="nsew")

        # Display editable entries
        for col, value in enumerate(row_values, start=1):
            entry = tk.Entry(root, width=10, validate="key", validatecommand=vcmd)
            entry.insert(0, str(value))
            entry.grid(row=row+1, column=col, padx=1, pady=1)
            row_entries.append(entry)

        entry_widgets.append(row_entries)
        submit_btn = tk.Button(root, text="Calculer l'allocation", command=on_submit, font=('Arial', 10, 'bold'), padx=10, pady=5)
        submit_btn.grid(row=len(inputs.keys()) + 2, column=2, columnspan=2, padx=10, pady=10)
        # Resource label + input

    resource_label = tk.Label(root, text="Heure d'étude total:", font=('Arial', 10))
    resource_label.grid(row=len(inputs.keys()) + 2, column=0, padx=5, pady=10, sticky='e')

    resource_entry.grid(row=len(inputs.keys()) + 2, column=1, padx=5, pady=10, sticky='w')
        # You can later access `entry_widgets[row][col]` to get data

    max_benefit_label.grid(row=len(inputs.keys()) + 3, column=0, columnspan=5, sticky='w', padx=10, pady=5)
    allocation_plan_label.grid(row=len(inputs.keys()) + 4, column=0, columnspan=5, sticky='w', padx=10, pady=5)
    Label(root, text="Heures", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)

    default_hours = [5, 10, 15, 20]
    for col, hour in enumerate(default_hours, start=1):
        entry = tk.Entry(root, width=10)
        entry.insert(0, str(hour))
        entry.grid(row=0, column=col, padx=1, pady=1)
        hour_entries.append(entry)

    # Actual header below editable hour row
    header_labels = ["Cours"]
    for col, text in enumerate(header_labels):
        label = tk.Label(root, text=text, font=('Arial', 10, 'bold'), borderwidth=1, relief="solid", padx=5, pady=5)
        label.grid(row=1, column=col, sticky="nsew")

    root.mainloop()

def grades_to_value(grade):
    return grades[grade]

if __name__ == "__main__":
    #resources = 50 // 5 # 10 block de 5h
    #t = map(gradesToValue, inputs["comptabilite"])
    """
    benefit_matrix = [
        list(map(gradesToValue, inputs["comptabilite"])),
        list(map(gradesToValue, inputs["finance"])),
        list(map(gradesToValue, inputs["science_decision"])),
        list(map(gradesToValue, inputs["gestion"])), # Cours 1
    ]
    """
    #print(benefit_matrix)
    #max_benefit, allocation_plan = resource_allocation_dp(resources, benefit_matrix, resources_matrix)
    #print("Maximum Benefit:", max_benefit)
    #print("Resource Allocation Plan:", allocation_plan)
    run_tkinter()