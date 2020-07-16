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
    print(f'type x_next: {type(x_next)}')
    # integrator
    F = Function('F', [x, u], x_next, \
                ['x', 'u'], ['x_next']);
    print(f'F: {F}')

    # NLP formulation
    # collect all decision variables in w
    g = [];
    # full state vector w: [u0, theta1, omega1, ....u49, theta50, omega50]
    w = [];
    # constraint on the entire state vector
    ubw = [];
    w0 = 0.1;
    L = 0;
    X_k = x0bar;

    for i in range(N):
        U_name = 'U_' + str(i)
        U_k = MX.sym(U_name, nu, 1);
        ubw.append(2)
        L = L + X_k[0]**2.0 + 0.1*(X_k[1])**2;
        L = L + 0.01 * U_k**2;

        X_next = F(X_k, U_k);

        X_name = 'X_' + str(i+1)
        X_k = MX.sym(X_name, nx, 1);
        ubw.append(inf) # no constraint on theta value
        ubw.append(8)   # omega max: 8
        w.append(U_k)
        w.append(X_k)
        g.append(X_next - X_k)
    L = L + 10*(X_k[0]**2.0 + X_k[1]**2.0);

    # create nlp solver
    # nlp = struct('x', vertcat(w{:}), 'f', L, 'g', g);
    nlp = {"x": w, "f": L, "g": g}
    solver = nlpsol('solver','ipopt', nlp);

    # solve nlp
    sol = solver('x0', w0, 'lbx', -ubw, 'ubx', ubw, 'lbg', 0, 'ubg', 0);
