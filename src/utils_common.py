import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, result_path, is_noc, show_plots = True):
        self.result_path = result_path
        self._showplots = show_plots
        self.is_noc = is_noc
        sns.set()

    def plot_state_trajectory(self, th, thdot):
        fig, ax = plt.subplots()
        ax.plot(th, label='angular displacement')
        ax.plot(thdot, label='angular velocity')
        ax.set_xlabel('timsteps N')
        ax.set_title('state trajectory')
        ax.legend(loc='upper right')

        #save the plot
        if (self.is_noc):
            fig.savefig(self.result_path + 'state_trajectory_noc.png')
        else:
            fig.savefig(self.result_path + 'state_trajectory_rl.png')

    def plot_dynamics(self, g_1, g_2):
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.plot(g_1, label='g_1')
        ax2.plot(g_2, label='g_2')
        ax2.set_xlabel('timsteps N')
        fig.suptitle('dynamics')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')

        if (self.is_noc):
            fig.savefig(self.result_path + 'dynamics_noc.png')
        else:
            fig.savefig(self.result_path + 'dynamics_rl.png')

    def plot_control_trajectory(self, u):
            fig, ax = plt.subplots()
            ax.plot(u, label='Control')
            ax.set_xlabel('timsteps N')
            ax.set_title('Control trajectory')
            ax.legend(loc='upper right')

            my_path = os.path.abspath(__file__)
            if (self.is_noc):
                fig.savefig(self.result_path + 'Control_trajectory_noc.png')
            else:
                fig.savefig(self.result_path + 'Control_trajectory_rl.png')

    def plot_costs(self, costs):
        fig, ax = plt.subplots()
        ax.plot(costs, label='Cost')
        ax.set_xlabel('timsteps N')
        ax.set_title('Cost trajectory')
        ax.legend(loc='upper right')

        #save the plot
        my_path = os.path.abspath(__file__)
        if (self.is_noc):
            fig.savefig(self.result_path + 'Cost_noc.png')
        else:
            fig.savefig(self.result_path + 'Cost_rl.png')