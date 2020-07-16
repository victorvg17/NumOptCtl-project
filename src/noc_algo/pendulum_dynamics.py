import numpy as np

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
        # load the arguements to thdot and th st eqn stays clean
        th = x[0]
        thdot = x[1]

        newthdot = thdot + (self.t1 * np.sin(th + np.pi) + self.t2*u) * self.dt + np.random.normal(loc=0.0, scale=0.01, size=None)
        newth = th + newthdot*self.dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        # newthdot = self.clip(newthdot)
        next_state = [newth, newthdot]
        return next_state

    def angle_normalize(self, theta):
        return (((theta+np.pi) % (2*np.pi)) - np.pi)
