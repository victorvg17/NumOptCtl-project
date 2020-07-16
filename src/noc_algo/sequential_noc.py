import numpy as np
from casadi import *
import math
from pendulum_dynamics import PendulumDynamics


if __name__ == '__main__':
    #parameters
    nx = 2;         # state dimension
    nu = 1;         # control dimension
    N = 50;         # horizon length
    DT = .1;        # discretization time step
    N_rk4 = 10;     # number of rk4 steps per discretiazion time step
    x0bar = [np.pi, 0];   # initial state
    h = DT / N_rk4; # integration step

    x = MX.sym('x',nx,1);
    u = MX.sym('u',nu,1);

    pend_dyn = PendulumDynamics()

    x_next = pend_dyn.simulate_next_state(x, u);
    # integrator
    F = Function('F', [x, u], [x_next]);
