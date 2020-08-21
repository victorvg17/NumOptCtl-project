import numpy as np
from casadi import *
import math
import matplotlib.pyplot as plt
from pendulum import PendulumEnv
import time
import importlib.util
import os


utils_path = os.path.abspath(__file__).replace(os.path.abspath(__file__).split("/")[-2] + '/' +os.path.abspath(__file__).split("/")[-1],'')
spec = importlib.util.spec_from_file_location("utils_common", utils_path + "utils_common.py")
plot_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_utils)
vis = plot_utils.Plotter(utils_path + 'results/', True)

def extractSolFromNlpSolver(res, iter_count):
    w_opt_np = np.array(res["x"])

    u_opt = []
    th_opt = []
    thdot_opt = []
    for i in range(0, w_opt_np.shape[0], 3):
        u_opt.append(w_opt_np[i])
        th_opt.append(w_opt_np[i+1])
        thdot_opt.append(w_opt_np[i+2])
    u_opt = np.array(u_opt)
    th_opt = np.array(th_opt)
    thdot_opt = np.array(thdot_opt)
    print('u_opt shape:', {u_opt.shape})
    print('th_opt shape:', {th_opt.shape})
    print('thdot_opt shape:', {thdot_opt.shape})

    cost = np.array(res["f"])
    loss = cost[0][0]
    g = np.array(res["g"])
    print('cost: ', loss)
    print('g shape:', {g.shape})
    g_1 = []
    g_2 = []
    for i in range(0, g.shape[0], 2):
        g_1.append(g[i])
        g_2.append(g[i+1])

    return u_opt, th_opt, thdot_opt, g_1, g_2, loss, iter_count


def simultaneous_opt(x0bar,L, pend_dyn, N , gamma):
    #parameters
    nx = pend_dyn.nx       # state dimension
    nu = pend_dyn.nu        # control dimension
    w0 = pend_dyn.w0
    max_speed = pend_dyn.max_speed
    max_torque = pend_dyn.max_torque

    x = MX.sym('x',nx,1)
    u = MX.sym('u',nu,1)
    x_next = pend_dyn.simulate_next_state(x, u)
    # print('type x_next:', type(x_next))
    # integrator
    F = Function('F', [x, u], [x_next], \
                ['x', 'u'], ['x_next'])
    print('F:', F)

    # NLP formulation
    # collect all decision variables in w
    g = []
    w = [] # full state vector w: [u0, theta1, omega1, ....u49, theta50, omega50]
    ubw = []  # constraint on the entire state vector
    X_k = x0bar

    for i in range(N):
        U_name = 'U_' + str(i)
        U_k = MX.sym(U_name, nu, 1)
        # L +=  X_k[0]**2 + 0.1*(X_k[1])**2 + 0.001*U_k**2
        L = X_k[0]**2 + 0.1*(X_k[1])**2 + 0.001*U_k**2 + gamma*L
        X_next = F(X_k, U_k)
        X_name = 'X_' + str(i+1)
        X_k = MX.sym(X_name, nx, 1)
        ubw = vertcat(ubw, max_torque, inf, max_speed)
        w = vertcat(w, U_k, X_k)
        g = vertcat(g, X_next - X_k)
    #L += 10*(X_k[0]**2.0 + X_k[1]**2.0)  #terminal cost

    # print the dimensions
    print("w shape:" ,np.shape(w), 'g shape:' ,np.shape(g), 'ubw shape:', np.shape(ubw))
    # create nlp solver
    nlp = {"x": w,
           "f": L,
           "g": g}
           
    opts_dict = {}
    opts_dict["ipopt.print_level"] = 0
    opts_dict["print_time"] = 0

    solver = nlpsol('solver','ipopt', nlp, opts_dict)

    arg = {}
    arg["x0"] = w0
    arg["lbx"] = -ubw
    arg["ubx"] = ubw
    arg["lbg"] = 0
    arg["ubg"] = 0

    # Solve the problem
    res = solver(**arg)
    it = solver.stats()['iter_count']
    return extractSolFromNlpSolver(res, it)

if __name__ == '__main__':
    
    N_rk4 = 10      # RK4 steps
    dt = 0.1        #delta time s
    N = 500         # horizon length
    gamma = 0.99

    costs = []
    u_opt_f = []
    th_opt_f = [] 
    thdot_opt_f = [] 
    g_1_f = []
    g_2_f = []
    iter_f = []
    iter_total = 0

    with contextlib.closing(PendulumEnv(N_rk4 = N_rk4, DT = dt)) as env:
        s = env.reset(fixed = True) #fixed start at pi,0
        x0bar = np.array(s)
        c = 0
        for i in range(N):
            env.render()
            u_opt, th_opt, thdot_opt, g_1 , g_2, l, iter_count = simultaneous_opt(s ,c ,env, N-i, gamma)
            s, c, d, _ = env.step(u_opt[0], noise = True) #state noise
            u_opt_f.append(u_opt[0])
            th_opt_f.append(th_opt[0])
            thdot_opt_f.append(thdot_opt[0])
            costs.append(c)  #gamma is implicit
            g_1_f.append(g_1[0])
            g_2_f.append(g_2[0])
            iter_f.append(iter_count)
            

        # visualise solution
    print("initial state:", x0bar)
    print("total iterations:", sum(iter_f))
    vis.plot_stats(iter_f, sum(iter_f))
    vis.plot_costs(costs)
    vis.plot_state_trajectory(th_opt_f, thdot_opt_f)
    vis.plot_control_trajectory(u_opt_f)
    vis.plot_dynamics(g_1_f, g_2_f)
