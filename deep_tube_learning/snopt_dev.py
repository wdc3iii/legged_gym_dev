import casadi as ca

# Define your problem (objective, constraints, etc.)
x = ca.MX.sym('x', 2)
obj = x[0]**2 + x[1]**2
g = x[0] + x[1] - 1

nlp = {'x': x, 'f': obj, 'g': g}

# Create the solver instance, specifying SNOPT
opts = {'nlpsol': 'snopt'}  # Ensure SNOPT is specified
solver = ca.nlpsol('solver', 'snopt', nlp, opts)

# Solve the problem
sol = solver(x0=[0.5, 0.5], lbg=0, ubg=0)

print(sol['x'])