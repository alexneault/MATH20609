from quasi_newton.py import quasi_newton  # Replace 'your_script_name' with the file containing your function

# Example inputs
inputs = {
    'fonction': ["x**2 - 4"], 
    'min': [1], 
    'max': [3], 
    'precision': [1e-6], 
    'newton': [None, None, 2], 
    'animationordinateur': [0]
}

# Create a dummy Excel workbook
wb = xw.Book()
ws = wb.sheets[0]

# Run the Quasi-Newton function
quasi_newton(inputs, ws, 1, 3)

# Print the result from the Excel sheet
result = ws.range("C2").value
print(f"Approximate root: {result}")