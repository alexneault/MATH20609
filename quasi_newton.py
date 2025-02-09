def quasi_newton(inputs: dict, ws: xw.Sheet, min, max):
    """
    Implements the Secant method (a Quasi-Newton approach) to find the root of a function.
    
    Parameters:
    inputs : dict
        Dictionary containing function details and parameters.
    ws : xw.Sheet
        Excel sheet where results will be stored.
    min : float
        Lower bound of the interval.
    max : float
        Upper bound of the interval.
    
    Returns:
    None (results are written to the Excel sheet).
    """
    secant_approxs = {}
    col_secant = inputs['newton'][2]
    func = inputs['fonction'][0]
    precision_required = inputs['precision'][0]

    x = sp.Symbol("x")
    func2 = sp.sympify(func)
    x1 = inputs['min'][0]
    x2 = inputs['max'][0]
    
    f = sp.lambdify(x, func2, 'math')
    
    precision_result = float('inf')
    secant_list = []
    max_iterations = 1000  
    iteration = 0
    
    if x1 * x2 > 0:
        secant_result = "Il n'y a pas de zéro dans l'intervalle donné de cette fonction"
    
    while precision_result > precision_required and iteration < max_iterations:
        fx1, fx2 = f(x1), f(x2)

        if fx2 - fx1 == 0:
            return "Erreur : Division par zéro, impossible d'effectuer l'approximation avec la méthode de la sécante."

        x_new = x2 - fx2 * ((x2 - x1) / (fx2 - fx1))
        precision_result = abs(x_new - x2)
        secant_list.append(x_new)
        secant_approxs[x_new] = f(x_new)
        
        x1, x2 = x2, x_new
        iteration += 1

    secant_result = x2 if precision_result <= precision_required else "Aucune convergence"
    
    ws.range(f"C{col_secant}").value = secant_result
    if inputs['animationordinateur'][0] == 1:
        add_animated_graph(secant_approxs, inputs, func, 'quasi-newton')