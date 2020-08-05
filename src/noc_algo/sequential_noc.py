import numpy as np
from casadi import *
import math
import matplotlib.pyplot as plt
from pendulum_dynamics import PendulumDynamics
from pendulum import PendulumEnv
import time


def extractSolFromNlpSolver(res):
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
    g = np.array(res["g"])
    print('cost: ', -cost[0][0])
    print('g shape:', {g.shape})
    g_1 = []
    g_2 = []
    for i in range(0, g.shape[0], 2):
        g_1.append(g[i])
        g_2.append(g[i+1])

    return u_opt, th_opt, thdot_opt, g_1, g_2

def calculateCostFromOptimTrajectory(u_opt, th_opt, thdot_opt):
    costs = []
    for u, th, thdot in zip(u_opt, th_opt, thdot_opt):
        curr_cost = th**2 + 0.1*thdot**2 + 0.001*u**2
        costs.append(curr_cost)
    costs = np.array(costs)
    print(f'costs shape: {costs.shape}')
    return costs

if __name__ == '__main__':
    #parameters
    nx = 2          # state dimension
    nu = 1          # control dimension
    N = 50         # horizon length
    N_rk4 = 10      # RK4 steps
    dt = 0.1        #delta time s
    x0bar = [np.pi, 0]    # initial state

    x = MX.sym('x',nx,1)
    u = MX.sym('u',nu,1)

    pend_dyn = PendulumDynamics(N_rk4 = N_rk4, DT = dt)
    max_speed = pend_dyn.max_speed
    max_torque = pend_dyn.max_torque
    x_next = pend_dyn.simulate_next_state(x, u)
    print('type x_next:', type(x_next))
    # integrator
    F = Function('F', [x, u], [x_next], \
                ['x', 'u'], ['x_next'])
    print('F:', F)

    # NLP formulation
    # collect all decision variables in w
    g = []
    # full state vector w: [u0, theta1, omega1, ....u49, theta50, omega50]
    w = []
    # constraint on the entire state vector
    ubw = []
    w0 = 0.1
    L = 0
    X_k = x0bar

    for i in range(N):
        U_name = 'U_' + str(i)
        U_k = MX.sym(U_name, nu, 1)
        L +=  X_k[0]**2 + 0.1*(X_k[1])**2 + 0.001*U_k**2

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
    solver = nlpsol('solver','ipopt', nlp)

    arg = {}
    arg["x0"] = w0
    arg["lbx"] = -ubw
    arg["ubx"] = ubw
    arg["lbg"] = 0
    arg["ubg"] = 0

    # Solve the problem
    res = solver(**arg)

    # visualise solution
    u_opt, th_opt, thdot_opt, g_1 , g_2 = extractSolFromNlpSolver(res)

    costs_opt = calculateCostFromOptimTrajectory(u_opt, th_opt, thdot_opt)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(th_opt)
    ax1.plot(thdot_opt)
    ax2.plot(u_opt)

    fig2 , (ax3, ax4) = plt.subplots(2,1)
    ax3.plot(g_1)
    ax4.plot(g_2)

    #env = PendulumEnv(N_rk4 = N_rk4, DT = dt)
    #state = []
    cost = 0
    list_cost = []
    with contextlib.closing(PendulumEnv(N_rk4 = N_rk4, DT = dt)) as env:
        s = env.reset()
        for i in range(N):
            env.render()
            s, c, d, _ = env.step(u_opt[i])
            #state.append(s)
            cost += c
            list_cost.append(c)
            time.sleep(dt)
    print("cost_rl: ", cost)
    fig3, ax5 = plt.subplots()
    ax5.plot(list_cost)
    plt.show()
    #plt.plot(state)
