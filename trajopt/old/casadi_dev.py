from casadi import *


class IterationCallback(Callback):
    def __init__(self, nlp, opts):
        Callback.__init__(self)
        self.construct("IterationCallback", opts)
        self.iteration_data = []

    def init(self):
        print("Initializing callback")

    def get_n_in(self):
        return 1  # Number of inputs

    def get_n_out(self):
        return 0  # Number of outputs

    def get_name_in(self, i):
        return 'i'

    def get_name_out(self, i):
        return 'ret'

    def eval(self, args):
        # Here, `args` contains the input arguments passed from the solver.
        # args[0] is the iteration number, args[1] is the objective function value, etc.
        iteration_number = args[0]
        objective_value = args[1]
        constraints = args[2]
        variables = args[3]

        # Store iteration data
        self.iteration_data.append([iteration_number, objective_value, constraints, variables])

        # Optionally, print or save the data to a file
        print(f"Iteration {iteration_number}: Objective = {objective_value}")


# Define the NLP problem
x = MX.sym('x', 2)
f = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
g = x[0] + x[1] - 1
nlp = {'x': x, 'f': f, 'g': g}

# Create the solver
opts = {
    'ipopt': {
        'print_level': 5,  # Print detailed iteration information
        'print_timing_statistics': 'yes',
    }
}
solver = nlpsol('solver', 'ipopt', nlp, opts)

# Instantiate the callback
callback = IterationCallback(nlp, {})

# Pass the callback to the solver's options
opts = {}
opts['iteration_callback'] = callback

# Solve the problem
sol = solver(x0=[2, 3], lbg=0, ubg=0, callback=callback)

# Access the iteration data stored in the callback
iteration_results = callback.iteration_data