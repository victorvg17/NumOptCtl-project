import numpy as np
from casadi import *

# build integrator dynamics
class PendulumDynamics:
    def __init__(self, DT = 0.1, N_rk4 = 10):
        self.max_speed = 8
        self.max_torque = 2.0
        self.DT = DT
        self.g = 9.8
        self.m = 1.0
        self.l = 1.0

        self.t1 = -3.0*self.g/(2*self.l)
        self.t2 = 3.0/(self.m*self.l**2)
        # types of integrator: implicit-euler and rk4
        self.kinematics_integrator = 'rk4'
        self.N_rk4 = N_rk4

    def clip(self, thdot):
        # in house clipping function for thdot output since Casadi variables cannot be dealt by numpy functions
        if (thdot < -self.max_speed):
            thdot = -self.max_speed
        elif (thdot > self.max_speed):
            thdot = self.max_speed
        return thdot

    def dynamics(self, x, u):
        theta = x[0]
        omega = x[1]
        return vertcat(omega, self.t1 * np.sin(theta + np.pi) + self.t2*u)

    def rk4step(self, x, u, h):
        # % one rk4 step
        # % inputs:
        # %  x             initial state of integration
        # %  u             control, kept constant over integration
        # %  h             time step of integration
        # % output:
        # %  x_next        state after one rk4 step
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + h/2*k1, u)
        k3 = self.dynamics(x+h/2*k2, u)
        k4 = self.dynamics(x+h*k3, u)
        x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def simulate_next_state(self, x, u):
        # newthdot = thdot + (self.t1 * np.sin(th + np.pi) + self.t2*u) * self.dt
        if (self.kinematics_integrator == 'implicit-euler'):
            theta = x[0]
            omega = x[1]
            new_omega = omega + (self.t1 * np.sin(theta + np.pi) + self.t2*u) * self.DT
            new_theta = theta + new_omega*self.DT
            return vertcat(new_theta, new_omega)

        else: #rk4 integrator
            h = self.DT/self.N_rk4
            for i in range(self.N_rk4):
                x = self.rk4step(x, u, h)

        return x
