import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib import path
from matplotlib.patches import Polygon
import cPickle as pickle

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
    def FW(event):
        self.ind = (self.ind + 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_fw = plt.axes([0.70, 0.80, 0.09, 0.02])
    self.b_fw = Button(self.ax_fw, 'F/W')
    self.b_fw.on_clicked(FW)

    # 3.2.2 play 1 b/w step
    def BW(event):
        self.ind = (self.ind - 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_bw = plt.axes([0.60, 0.80, 0.09, 0.02])
    self.b_bw = Button(self.ax_bw, 'B/W')
    self.b_bw.on_clicked(BW)

    # 4. Hand-Craft Features
    def set_color_by_cond(event):
        self.update_cond_vector(self)
        self.tsne_scat.set_array(self.cond_vector)
        sizes = 5 * self.cond_vector + self.pnt_size * np.ones_like(self.tsne_scat.get_sizes())
        self.tsne_scat.set_sizes(sizes)
        print ('number of valid points: %d') % (np.sum(self.cond_vector))

    self.ax_cond = plt.axes([0.70, 0.77, 0.09, 0.02])
    self.cond_vector = np.ones(shape=(self.num_points,1), dtype='int8')
    self.b_color_by_cond = Button(self.ax_cond, 'color by cond')
    self.b_color_by_cond.on_clicked(set_color_by_cond)

    self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

    # 5. mark cluster
    def mark_cluster(event):
        marked_points = plt.ginput(0)
        poly_path = path.Path(marked_points)
        cluster_points = poly_path.contains_points(self.data_t.T)
        sizes = 5 * cluster_points + self.pnt_size * np.ones_like(self.num_points)
        self.tsne_scat.set_sizes(sizes)
        self.ax_tsne.add_patch(Polygon(marked_points, alpha=0.2))
        # update cluster points
        self.cluster_number += 1
        self.cluster_ids[cluster_points] = self.cluster_number
        # annotate cluster
        self.ax_tsne.annotate(self.cluster_number, xy=marked_points[0])
        # save cluster ids vector
        pickle.dump(self.cluster_ids, file(self.global_feats['data_dir'] + 'cluster_ids.bin','wb'))

    self.ax_mc = plt.axes([0.60, 0.77, 0.09, 0.02])
    self.b_mc = Button(self.ax_mc, 'Mark Cluster')
    self.b_mc.on_clicked(mark_cluster)
    self.cluster_ids = np.zeros(self.num_points)
    self.cluster_number = 0

    self.SLIDER_FUNCS = []
    self.CHECK_BUTTONS = []
