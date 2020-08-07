import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self, result_path, show_plots = True):
        self.result_path = result_path
        self._showplots = show_plots
        sns.set()

    def plot_state_trajectory(self, th, thdot, is_noc = True):
        fig, ax = plt.subplots()
        ax.plot(th, label='angular displacement')
        ax.plot(thdot, label='angular velocity')
        ax.set_xlabel('timsteps N')
        ax.set_title('state trajectory')
        ax.legend(loc='upper right')

        #save the plot
        my_path = os.path.abspath(__file__)
        if (is_noc):
            fig.savefig(my_path + '/results/state_trajectory_noc.png')
        else:
            fig.savefig(my_path + '/results/state_trajectory_rl.png')
