import numpy as np
from casadi import *

# build integrator dynamics
class PendulumDynamics:
    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = 10.
        self.m = 1.
        self.l = 1.

        self.t1 = -3*self.g/(2*self.l)
        self.t2 = 3./(self.m*self.l**2)

    def clip(self, thdot):
        # in house clipping function for thdot output since Casadi variables cannot be dealt by numpy functiona
        if (thdot < -self.max_speed):
            thdot = -self.max_speed
        elif (thdot > self.max_speed):
            thdot = self.max_speed
        return thdot

    def simulate_next_state(self, x, u):
        # newthdot = thdot + (self.t1 * np.sin(th + np.pi) + self.t2*u) * self.dt
        newthdot = x[1] + (self.t1 * np.sin(x[0] + np.pi) + self.t2*u) * self.dt

        newth = x[0] + newthdot*self.dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        # X_new = MX.sym('X_new', 2, 1)
        # X_new[0] = newth
        # X_new[1] = newthdot

        return vertcat(newth, newthdot)
