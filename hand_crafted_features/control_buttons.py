import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib import path
from matplotlib.patches import Polygon
import common

def add_buttons(self):

    #############################
    # 1. play 1 b/w step
    #############################
    def BW(event):
        self.ind = (self.ind - 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_bw = plt.axes([0.60, 0.80, 0.09, 0.02])
    self.b_bw = Button(self.ax_bw, 'B/W')
    self.b_bw.on_clicked(BW)

    #############################
    # 2. play 1 f/w step
    #############################
    def FW(event):
        self.ind = (self.ind + 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_fw = plt.axes([0.70, 0.80, 0.09, 0.02])
    self.b_fw = Button(self.ax_fw, 'F/W')
    self.b_fw.on_clicked(FW)

    #############################
    # 3. Color by condition
    #############################
    def set_color_by_cond(event):
        self.update_cond_vector(self)
        self.tsne_scat.set_array(self.cond_vector)
        sizes = 5 * self.cond_vector + self.pnt_size * np.ones_like(self.tsne_scat.get_sizes())
        self.tsne_scat.set_sizes(sizes)
        print ('number of valid points: %d') % (np.sum(self.cond_vector))

    self.ax_cond = plt.axes([0.60, 0.77, 0.09, 0.02])
    self.cond_vector = np.ones(shape=(self.num_points,1), dtype='int8')
    self.b_color_by_cond = Button(self.ax_cond, 'color by cond')
    self.b_color_by_cond.on_clicked(set_color_by_cond)

    self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

    #############################
    # 4. save figure
    #############################
    def save_figure(event):
        plt.savefig(self.global_feats['data_dir'] + '/knn/tsne_figure.png')
        print 'saved figure to %s' %  (self.global_feats['data_dir'] + '/knn/tsne_figure.png')

    self.ax_save = plt.axes([0.70, 0.77, 0.09, 0.02])
    self.b_save = Button(self.ax_save, 'Save figure')
    self.b_save.on_clicked(save_figure)

    #############################
    # 5. mark cluster
    #############################
    def mark_cluster(event):
        # user marks cluster area
        marked_points = plt.ginput(0)

        # create path object from marked points
        poly_path = path.Path(marked_points)

        # recieve the points that are inside the marked area
        cluster_points = poly_path.contains_points(self.data_t.T)

        # update the list of polygons
        self.cluster_polygons.append(Polygon(marked_points, alpha=0.2))

        # draw new cluster
        self.ax_tsne.add_patch(self.cluster_polygons[-1])

        # update cluster points
        self.cluster_number += 1
        self.cluster_ids[cluster_points] = self.cluster_number

        # annotate cluster
        self.cluster_annotations.append(self.ax_tsne.annotate(self.cluster_number, xy=marked_points[0], size=20, color='r'))

        # save cluster ids vector
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.cluster_ids)

    self.ax_mc = plt.axes([0.60, 0.74, 0.09, 0.02])
    self.b_mc = Button(self.ax_mc, 'Mark Cluster')
    self.b_mc.on_clicked(mark_cluster)
    self.cluster_ids = np.zeros(self.num_points)
    self.cluster_polygons = []
    self.cluster_annotations = []
    self.cluster_number = 0

    #############################
    # 6. delete cluster
    #############################
    def delete_cluster(event):
        # remove cluster points
        self.cluster_ids[self.cluster_ids==self.cluster_number] = 0

        # decrease cluster number by 1
        self.cluster_number -= 1

        # delete cluster from figure
        self.cluster_polygons[-1].remove()
        self.cluster_annotations[-1].remove()

        # delete cluster from list
        self.cluster_polygons.pop()
        self.cluster_annotations.pop()

        # re-save cluster ids
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.cluster_ids)

    self.ax_dc = plt.axes([0.70, 0.74, 0.09, 0.02])
    self.b_dc = Button(self.ax_dc, 'Delete Cluster')
    self.b_dc.on_clicked(delete_cluster)