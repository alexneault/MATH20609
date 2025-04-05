import sympy as sp


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