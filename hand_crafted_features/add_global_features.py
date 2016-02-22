import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons

def add_buttons(self, global_feats):
    # 3.1 global coloring buttons
    self.COLORS = {}
    self.add_color_button([0.60, 0.95, 0.09, 0.02], 'value', global_feats['value'])
    self.add_color_button([0.70, 0.95, 0.09, 0.02], 'actions', global_feats['actions'])
    self.add_color_button([0.60, 0.92, 0.09, 0.02], 'termination', global_feats['termination'])
    self.add_color_button([0.70, 0.92, 0.09, 0.02], 'time', global_feats['time'])
    self.add_color_button([0.60, 0.89, 0.09, 0.02], 'risk', global_feats['risk'])
    self.add_color_button([0.70, 0.89, 0.09, 0.02], 'TD', global_feats['TD'])
    self.add_color_button([0.60, 0.86, 0.09, 0.02], 'action repetition', global_feats['act_rep'])
    self.add_color_button([0.70, 0.86, 0.09, 0.02], 'reward', global_feats['reward'])

    # 3.2 global control buttons

    # 3.2.1 play 1 f/w step
    self.ax_fw = plt.axes([0.70, 0.80, 0.09, 0.02]) # button : forward
    self.b_fw = Button(self.ax_fw, 'F/W')
    self.b_fw.on_clicked(self.FW)

    # 3.2.2 play 1 b/w step
    self.ax_bw = plt.axes([0.60, 0.80, 0.09, 0.02]) # button : backward
    self.b_bw = Button(self.ax_bw, 'B/W')
    self.b_bw.on_clicked(self.BW)

    # 4. Hand-Craft Features
    self.ax_cond = plt.axes([0.65, 0.73, 0.09, 0.02]) # button : color by condition
    self.cond_vector = np.ones(shape=(self.num_points,1), dtype='int8')
    self.b_color_by_cond = Button(self.ax_cond, 'color by cond')
    self.b_color_by_cond.on_clicked(self.set_color_by_cond)

    self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

    self.SLIDER_FUNCS = []
    self.CHECK_BUTTONS = []
