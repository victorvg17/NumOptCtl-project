import numpy as np
from casadi import *

# build integrator dynamics
class PendulumDynamics:
    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.1
        self.g = 9.8
        self.m = 1.
        self.l = 1.

        self.t1 = -3*self.g/(2*self.l)
        self.t2 = 3./(self.m*self.l**2)

    def clip(self, thdot):
        # in house clipping function for thdot output since Casadi variables cannot be dealt by numpy functions
        if (thdot < -self.max_speed):
            thdot = -self.max_speed
        elif (thdot > self.max_speed):
            thdot = self.max_speed
        return thdot

    def simulate_next_state(self, x, u):
        # newthdot = thdot + (self.t1 * np.sin(th + np.pi) + self.t2*u) * self.dt
        newthdot = x[1] + (self.t1 * sin(x[0] + np.pi) + self.t2*u) * self.dt
        newth = x[0] + newthdot*self.dt
        return vertcat(newth, newthdot)
