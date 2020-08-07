import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.8, DT = 0.1, N_rk4 = 10, integrator = 'rk4'):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = DT
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        self.t1 = -3.0*self.g/(2*self.l)
        self.t2 = 3.0/(self.m*self.l**2)

        self.kinematics_integrator = integrator
        self.N_rk4 = N_rk4

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, x, u):
        theta = x[0]
        omega = x[1]
        dyn_eqn = np.array([omega, self.t1 * np.sin(theta) + self.t2*u])
        return dyn_eqn

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

        # a_max = np.array([np.pi, self.max_speed])
        # x_next[1] += np.random.normal(loc=0.0, scale=0.01, size=None)
        # x_next = np.clip(x_next, a_min = -a_max, a_max = a_max)

        return x_next

    def step(self,u):
        th, thdot = self.state

        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        #costs = angle_normalize(th)**2 + 0.1*thdot**2 + 0.001*(u**2)
        costs = th**2 + 0.1*thdot**2 + 0.001*(u**2)
        if (self.kinematics_integrator == 'implicit-euler'):
            newthdot = thdot + (self.t1 * np.sin(th + np.pi) + self.t2*u) * dt \
                            + np.random.normal(loc=0.0, scale=0.01, size=None)
            newth = th + newthdot*dt
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
            self.state = np.array([newth, newthdot])
        else: #rk4 integrator
            h = dt/self.N_rk4
            for i in range(self.N_rk4):
                self.state = self.rk4step(self.state, u, h)

        # return self._get_obs(), -costs, False, {}
        return self._get_obs(), costs, False, {}

    def reset(self, to_state):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.last_u = None
        self.state = np.array(to_state)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
        return np.array([theta, thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            # fname='/home/srbh/noc_proj/NumOptCtl-project/src/noc_algo/clockwise.png'
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
