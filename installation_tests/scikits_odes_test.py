import matplotlib.pyplot as plt
import numpy as np
from scikits.odes import ode

t0, y0 = 1, np.array([0.5, 0.5])  # initial condition
def van_der_pol(t, y, ydot):
    """ we create rhs equations for the problem"""
    ydot[0] = y[1]
    ydot[1] = 1000*(1.0-y[0]**2)*y[1]-y[0]

solution = ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
plt.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
plt.show()