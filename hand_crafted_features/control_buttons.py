import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import path
from matplotlib.patches import Polygon
from clustering import clustering_
import common
import pickle
from digraph import draw_transition_table

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

        # save figure
        plt.savefig(self.global_feats['data_dir'] + '/knn/tsne_figure.png')

        # save clusters
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.clusters['cluster_ids'])

        # save clusters (pickle)
        pickle.dump(self.clusters, file((self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'),'w'))

        print 'saved figure to %s' %  (self.global_feats['data_dir'] + '/knn/tsne_figure.png')

    self.ax_save = plt.axes([0.70, 0.77, 0.09, 0.02])
    self.b_save = Button(self.ax_save, 'Save figure')
    self.b_save.on_clicked(save_figure)

    #############################
    # 5.0 update cluster (helper function)
    #############################
    def update_cluster(self, marked_points):
        # create path object from marked points
        poly_path = path.Path(marked_points)

        # recieve the points that are inside the marked area
        cluster_points = poly_path.contains_points(self.data_t.T)

        # update the list of polygons
        self.clusters['polygons'].append(Polygon(marked_points, alpha=0.2))

        # draw new cluster
        self.ax_tsne.add_patch(self.clusters['polygons'][-1])

        # update cluster points
        self.clusters['cluster_number'] += 1

        self.clusters['cluster_ids'][cluster_points] = self.clusters['cluster_number']

        # update marked_points
        self.clusters['marked_points'].append(marked_points)

        # annotate cluster
        self.clusters['annotations'].append(self.ax_tsne.annotate(self.clusters['cluster_number'], xy=marked_points[0], size=20, color='r'))

    #############################
    # 5. mark cluster
    #############################
    def mark_cluster(event):
        # user marks cluster area
        marked_points = plt.ginput(0, timeout=-1)

        update_cluster(self, marked_points)

        # save cluster ids to hdf5 vector
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.clusters['cluster_ids'])

        # save clusters (pickle)
        pickle.dump(self.clusters, file((self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'),'w'))

    self.ax_mc = plt.axes([0.60, 0.74, 0.09, 0.02])
    self.b_mc = Button(self.ax_mc, 'Mark Cluster')
    self.b_mc.on_clicked(mark_cluster)
    self.clusters = {
            'polygons' : [],
            'annotations' : [],
            'cluster_ids' : np.zeros(self.num_points),
            'cluster_number' : 0,
            'cluster_points' : [],
            'marked_points' : [],
    }

    #############################
    # 6. delete cluster
    #############################
    def delete_cluster(event):
        # remove cluster points
        self.clusters['cluster_ids'][self.clusters['cluster_ids']==self.clusters['cluster_number']] = 0

        # decrease cluster number by 1
        self.clusters['cluster_number'] -= 1

        # delete cluster from figure
        self.clusters['polygons'][-1].remove()
        self.clusters['annotations'][-1].remove()

        # delete cluster from list
        self.clusters['polygons'].pop()
        self.clusters['annotations'].pop()
        self.clusters['marked_points'].pop()

    self.ax_dc = plt.axes([0.70, 0.74, 0.09, 0.02])
    self.b_dc = Button(self.ax_dc, 'Delete Cluster')
    self.b_dc.on_clicked(delete_cluster)

    #############################
    # 6. load clusters
    #############################
    def load_clusters(event):
        self.clusters = pickle.load(file(self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'))

        marked_points_list = self.clusters['marked_points']

        self.clusters = {
            'polygons' : [],
            'annotations' : [],
            'cluster_ids' : np.zeros(self.num_points),
            'cluster_number' : 0,
            'cluster_points' : [],
            'marked_points' : [],
        }

        # draw clusters
        for marked_points in marked_points_list:
            update_cluster(self, marked_points)

    self.ax_lc = plt.axes([0.60, 0.71, 0.09, 0.02])
    self.b_lc = Button(self.ax_lc, 'Load Clusters')
    self.b_lc.on_clicked(load_clusters)

    #############################
    # 7. cluster states
    #############################
    def clustering(event):
        clustering_(self,plt)

    self.ax_clstr = plt.axes([0.70, 0.71, 0.09, 0.02])
    self.b_clstr = Button(self.ax_clstr, 'Clustering')
    self.clustering_labels = None
    self.b_clstr.on_clicked(clustering)

    #############################
    # 8. Mark trajectory
    #############################
    def mark_trajectory(event):

        # 1. Display trajectory points over t-SNE
        traj_point_mask = np.asarray(self.hand_craft_feats['traj'])==self.traj_id
        self.tsne_scat.set_array(traj_point_mask)
        sizes = 5 * traj_point_mask + self.pnt_size * np.ones(self.num_points)
        self.tsne_scat.set_sizes(sizes)

        # 2. Display trajectory moves over SMDP
        if self.clustering_labels is not None:
            traj = self.traj_list[self.traj_id]
            moves = traj['moves']
            R = traj['R']
            length = traj['length']
            traj_moves = list(set(moves))
            draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=self.smdp.edges, red_edges=traj_moves)

            print ('Trajectory id: %d, Number of points: %d, R: %d') % (self.traj_id, length, R)

        self.traj_id = (self.traj_id + 1) % self.hand_craft_feats['n_trajs']

    self.traj_id = 0
    self.ax_mt = plt.axes([0.80, 0.80, 0.09, 0.02])
    self.b_mt = Button(self.ax_mt, 'Mark Traj.')
    self.b_mt.on_clicked(mark_trajectory)

    #############################
    # 9. Policy improvement
    #############################
    def policy_improvement(event):
        if self.clustering_labels is None:
            return

        policy = self.smdp.greedy_policy
        draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=None, red_edges=policy)

    self.ax_pi = plt.axes([0.80, 0.77, 0.09, 0.02])
    self.b_pi = Button(self.ax_pi, 'Policy improve.')
    self.b_pi.on_clicked(policy_improvement)
