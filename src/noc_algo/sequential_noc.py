import numpy as np
from casadi import *
import math

# build integrator dynamics
def dynamics(x, u):
    return [x(2),
            sin(x(1)) + u]


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

    x_next = x;
    for i = 1:N_rk4
        x_next = rk4step(x_next, u, dynamics, h);
    end
