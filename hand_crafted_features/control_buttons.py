import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib import path
from matplotlib.patches import Polygon
import common
import pickle
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import KMeans_st as Kmeans_st
from sklearn import preprocessing

from agglomerative_clustering import agg_clustering



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
    def clustering(N=20,W=1):
        import scipy.stats
        from digraph import draw_transition_table
        if self.clustering_labels is None:
            N = self.cluster_params['n_clusters']
            W = self.cluster_params['window_size']

            if self.cluster_params['method'] == 0: # 'kmeans'

                # 1. create data for clustering
                # 1.1 choose features
                n_features=5
                data = np.zeros(shape=(self.global_feats['tsne'].shape[0],n_features))
                data[:,0:2] = self.global_feats['tsne']
                data[:,2] = self.global_feats['value']
                data[:,3] = self.global_feats['time']
                data[:,4] = self.global_feats['termination']
                # data[:,5] = self.global_feats['tsne3d_norm']
                # data[:,6] = self.hand_craft_feats['missing_bricks']
                # data[:,6] = self.hand_craft_feats['hole']
                # data[:,7] = self.hand_craft_feats['racket']
                # data[:,8] = self.hand_craft_feats['ball_dir']
                # data[:,9] = self.hand_craft_feats['traj']
                # data[:,9:11] = self.hand_craft_feats['ball_pos']
                data[np.isnan(data)] = 0
                # 1.2 data standartization
                scaler = preprocessing.StandardScaler().fit(data)
                data = scaler.fit_transform(data)
                # data_scale = data.std(axis=0)
                # data_mean  = data.mean(axis=0)
                # data -= data_mean
                # data /= data_scale
                # 2. Build cluster model

                # 2.1 regular K-means
                # cluster_model = KMeans(n_clusters=N)

                # 2.2 entropy reducing k-means
                cluster_model = Kmeans_st(n_clusters=N,window_size=W,n_jobs=8,n_init=64)

                # 2.3 agglomerative clustering
                # cluster_model = agg_clustering(X=data, termination=self.global_feats['termination'], min_clusters=N, max_entropy=1)
                # cluster_model = AgglomerativeClustering(n_clusters=N)

                # 3. cluster points
                cluster_model.fit(data)
                ids = (cluster_model.labels_).astype(np.float32)
                self.clustering_labels = ids/ids.max()

                # 4. create transition matrix
                transition_table = np.zeros(shape=(N,N))
                labels = cluster_model.labels_
                for i in range(labels.shape[0]-1):
                    if labels[i+1]==labels[i]:
                        continue
                    else:
                        transition_table[labels[i],labels[i+1]] += 1
                self.transition_table = transition_table
                pickle.dump(self.transition_table, file((self.global_feats['data_dir'] + '/knn/' + 'transition_table.bin'),'w'))
                # pickle.dump(cluster_model.cluster_centers_, file((self.global_feats['data_dir'] + '/knn/' + 'cluster_centers.bin'),'w'))
                # pickle.dump(cluster_model.cluster_variance, file((self.global_feats['data_dir'] + '/knn/' + 'cluster_variance.bin'),'w'))

                self.transition_entropy = scipy.stats.entropy(self.transition_table.T).mean()
                # self.inertia = cluster_model.inertia_
                # print 'entropy: %f' % self.transition_entropy
                # print 'inertia: %f' % self.inertia

                ###### draw cluster indices
                plt.figure(self.fig.number)
                cluster_centers = scaler.inverse_transform(cluster_model.cluster_centers_)
                # cluster_centers*=data_scale
                # cluster_centers+=data_mean
                for i in range(N):
                    x = cluster_centers[i,0]
                    y = cluster_centers[i,1]
                    self.ax_tsne.annotate(i, xy=[x,y], size=20, color='r')
                draw_transition_table(self.transition_table,cluster_centers)
                plt.show()


            elif self.cluster_params['spectral']: #'spectral_clustering'
                import scipy.spatial.distance
                import scipy.sparse
                dists = scipy.spatial.distance.pdist(self.global_feats['tsne'], 'euclidean')
                similarity = np.exp(-dists/10)
                similarity[similarity<1e-2] = 0
                print 'Created similarity matrix'
                affine_mat = scipy.spatial.distance.squareform(similarity)
                # affine_mat = scipy.sparse.csr_matrix(affine_mat)
                cluster_model = SpectralClustering(n_clusters=N,affinity='precomputed')
                ids = cluster_model.fit_predict(affine_mat).astype(np.float32)
                self.clustering_labels = ids/ids.max()
        else:
            self.tsne_scat.set_array(self.clustering_labels)

    self.ax_clstr = plt.axes([0.80, 0.80, 0.09, 0.02])
    self.b_clstr = Button(self.ax_clstr, 'Clustering')
    self.clustering_labels = None
    self.b_clstr.on_clicked(clustering)

    ####################################
    # clustering grid search
    ####################################

    # res = np.zeros(shape=(36,4))
    # t=0
    # for k in range(5,31,5):
    #     for w in range(1,12,2):
    #         self.cluster_params['window_size'] = w
    #         self.cluster_params['n_clusters'] = k
    #         self.clustering_labels = None
    #         clustering()
    #         print 'entropy: %f' % self.transition_entropy + ' inertia: %f' % self.inertia + ' num_clusters: %f' %self.cluster_params['n_clusters'] + ' window_size: %f' %self.cluster_params['window_size']
    #         res[t,0]=w
    #         res[t,1]=k
    #         res[t,2]=self.transition_entropy
    #         res[t,3]=self.inertia
    #         t+=1
    # plt.figure("""first figure""") # Here's the part I need
    # line_entropy, = plt.plot(res[:,2]/res[:,2].max(),label='entropy')
    # line_inertia, = plt.plot(res[:,3]/res[:,3].max(),label='inertia')
    # line_sum, = plt.plot(res[:,3]/res[:,3].max()+res[:,2]/res[:,2].max(),label='sum')
    # plt.legend(handles=[line_entropy, line_inertia,line_sum])
    #
    # plt.show()
    #
    # print(res[2:4])

    #############################
    # 8. color outliers
    #############################
    def outliers(event):
        if self.outlier_color is None:
            # run your algorithm once
            from sos import sos
            import argparse
            import sys
            parser = argparse.ArgumentParser(description="Stochastic Outlier Selection")
            parser.add_argument('-b', '--binding-matrix', action='store_true',
            default=False, help="Print binding matrix", dest="binding_matrix")
            parser.add_argument('-t', '--threshold', type=float, default=None,
            help=("Float between 0.0 and 1.0 to use as threshold for selecting "
                "outliers. By default, this is not set, causing the outlier "
                "probabilities instead of the classification to be outputted"))
            parser.add_argument('-d', '--delimiter', type=str, default=',', help=(
            "String to use to separate values. By default, this is a comma."))
            parser.add_argument('-i', '--input', type=argparse.FileType('rb'),
            default=sys.stdin, help=("File to read data set from. By default, "
                "this is <stdin>."))
            parser.add_argument('-m', '--metric', type=str, default='euclidean', help=(
            "String indicating the metric to use to compute the dissimilarity "
            "matrix. By default, this is 'euclidean'. Use 'none' if the data set "
            "is a dissimilarity matrix."))
            parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
            default=sys.stdout, help=("File to write the computed outlier "
                "probabilities to. By default, this is <stdout>."))
            parser.add_argument('-p', '--perplexity', type=float, default=30.0,
            help="Float to use as perpexity. By default, this is 30.0.")
            parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help="Print debug messages to <stderr>.")
            args = parser.parse_args()
            self.outlier_color = sos(self.global_feats['tsne'], 'euclidean', 50,args)



        self.tsne_scat.set_array(self.outlier_color)
        sizes = np.ones(self.num_points)*self.pnt_size
        sizes[self.outlier_color>self.slider_outlier_thresh.val] = 250

        self.tsne_scat.set_sizes(sizes)
        self.fig.canvas.draw()



    def update_slider(self, name, slider):
        def f():
            setattr(self, name, slider.val)
        return f
    self.ax_otlyr = plt.axes([0.80, 0.77, 0.09, 0.02])
    self.b_otlyr = Button(self.ax_otlyr, 'Outliers')
    self.outlier_color = None
    self.b_otlyr.on_clicked(outliers)
    self.slider_outlier_thresh = Slider(plt.axes([0.80, 0.74, 0.09, 0.02]), 'outlier_thresh', valmin=0, valmax=1, valinit=0.75)
    self.SLIDER_FUNCS.append(update_slider(self, 'outlier_thresh', self.slider_outlier_thresh))
    self.slider_outlier_thresh.on_changed(self.update_sliders)
#