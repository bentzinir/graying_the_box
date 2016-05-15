from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import KMeans_st as Kmeans_st

from emhc import EMHC
from smdp import SMDP
import numpy as np
import common
from digraph import draw_transition_table

def calc_cluster_im(self,indices):
    screens = np.copy(self.screens[indices])
    if self.game_id  == 2: #pacman
        for s in screens:

            enemies_map = 1 * (s[:,:,0] == 180) + \
                          1 * (s[:,:,0] == 149) + \
                          1 * (s[:,:,0] == 212) + \
                          1 * (s[:,:,0] == 128) + \
                          1 * (s[:,:,0] == 232) + \
                          1 * (s[:,:,0] == 204)

            enemies_mask = np.ones((210,160),dtype=bool)
            enemies_mask[20:28,6:10] = 0
            enemies_mask[140:148,6:10] = 0
            enemies_mask[20:28,150:154] = 0
            enemies_mask[140:148,150:154] = 0
            enemies_map = enemies_map * enemies_mask
            r_ch = s[:,:,0]
            g_ch = s[:,:,1]
            b_ch = s[:,:,2]
            r_ch[np.nonzero(enemies_map)] = 45
            g_ch[np.nonzero(enemies_map)] = 50
            b_ch[np.nonzero(enemies_map)] = 184
    meanscreen=np.mean(screens,axis=0)

    return meanscreen

def draw_skills(self,cluster_ind,plt):
    state_indices = (self.clustering_labels==cluster_ind)
    cluster_mean_screen = calc_cluster_im(self,state_indices)

    plt.figure('Cluster %d skills' %cluster_ind)
    ax = plt.subplot(442)
    ax.imshow(cluster_mean_screen)
    ax.set_title('Full cluster %d' % cluster_ind)
    ax.axis('off')
    for i,l in enumerate(self.smdp.skill_indices[cluster_ind]):
        skill_mean_screen = calc_cluster_im(self,l)
        subplot_ind = 445+i
        ax = plt.subplot(subplot_ind)
        ax.imshow(skill_mean_screen)
        ax.set_title('Skill %d' %self.smdp.skill_list[cluster_ind][i])
        ax.axis('off')


def perpare_features(self, n_features):

    data = np.zeros(shape=(self.global_feats['tsne'].shape[0],n_features))
    data[:,0:2] = self.global_feats['tsne']
    data[:,2] = self.global_feats['value']
    # data[:,3] = self.global_feats['time']
    # data[:,4] = self.global_feats['termination']
    # data[:,5] = self.global_feats['tsne3d_norm']
    # data[:,6] = self.hand_craft_feats['missing_bricks']
    # data[:,6] = self.hand_craft_feats['hole']
    # data[:,7] = self.hand_craft_feats['racket']
    # data[:,8] = self.hand_craft_feats['ball_dir']
    # data[:,9] = self.hand_craft_feats['traj']
    # data[:,9:11] = self.hand_craft_feats['ball_pos']
    data[np.isnan(data)] = 0
    # 1.2 data standartization
    # scaler = preprocessing.StandardScaler(with_centering=False).fit(data)
    # data = scaler.fit_transform(data)
    data_scale = data.max(axis=0)
    # data_mean  = data.mean(axis=0)
    # data -= data_mean
    data /= data_scale
    return data, data_scale

def clustering_(self,plt):
    if self.clustering_labels is not None:
        self.tsne_scat.set_array(self.clustering_labels/self.clustering_labels.max())
        draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=self.smdp.edges)
        plt.show()
        return

    n_clusters = self.cluster_params['n_clusters']
    W = self.cluster_params['window_size']
    n_iters = self.cluster_params['n_iters']
    entropy_iters = self.cluster_params['entropy_iters']
    term = self.global_feats['termination']
    reward = self.global_feats['reward']
    value = self.global_feats['value']

    # 1. create data for clustering
    data, data_scale = perpare_features(self,n_features=3)

    # 2. Build cluster model
    # 2.1 spatio-temporal K-means
    if self.cluster_params['method'] == 0:
        windows_vec = np.arange(start=W,stop=W+1,step=1)
        clusters_vec = np.arange(start=n_clusters,stop=n_clusters+1,step=1)
        models_vec = []
        scores = np.zeros(shape=(len(clusters_vec),1))
        for i,n_w in enumerate(windows_vec):
            for j,n_c in enumerate(clusters_vec):
                cluster_model = Kmeans_st(n_clusters=n_clusters,window_size=n_w,n_jobs=8,n_init=n_iters,entropy_iters=entropy_iters)
                cluster_model.fit(data, rewards=reward, termination=term, values=value)
                labels = cluster_model.labels_
                models_vec.append(cluster_model.smdp)
                scores[j] = cluster_model.smdp.score
                print 'window size: %d , Value mse: %f' % (n_w, cluster_model.smdp.score)
            best = np.argmin(scores)
            self.cluster_params['n_clusters'] +=best
            self.smdp = models_vec[best]

    # 2.1 Spectral clustering
    elif self.cluster_params['method'] == 1:
        import scipy.spatial.distance
        import scipy.sparse
        dists = scipy.spatial.distance.pdist(self.global_feats['tsne'], 'euclidean')
        similarity = np.exp(-dists/10)
        similarity[similarity<1e-2] = 0
        print 'Created similarity matrix'
        affine_mat = scipy.spatial.distance.squareform(similarity)
        cluster_model = SpectralClustering(n_clusters=n_clusters,affinity='precomputed')
        labels = cluster_model.fit_predict(affine_mat)

    # 2.2 EMHC
    elif self.cluster_params['method'] == 2:
        # cluster with k means down to n_clusters + D
        n_clusters_ = n_clusters + 5
        kmeans_st_model = Kmeans_st(n_clusters=n_clusters_,window_size=W,n_jobs=8,n_init=n_iters,entropy_iters=entropy_iters, random_state=123)
        kmeans_st_model.fit(data, rewards=reward, termination=term, values=value)
        cluster_model = EMHC(X=data, labels=kmeans_st_model.labels_, termination=term, min_clusters=n_clusters, max_entropy=np.inf)
        cluster_model.fit()
        labels = cluster_model.labels_
        self.smdp = SMDP(labels=labels, termination=term, rewards=reward, values=value, n_clusters=n_clusters)

    self.clustering_labels = (labels).astype(np.float32)
    self.smdp.complete_smdp()
    common.create_trajectory_data(self)
    self.state_pi_correlation = common.reward_policy_correlation(self.traj_list, self.smdp.greedy_policy, self.smdp)

    # common.extermum_trajs_discrepency(self.traj_list)

    # for i in xrange(self.cluster_params['n_clusters']):
    #     draw_skills(self,i,plt)
    #     common.draw_skill_time_dist(self,i)
    common.visualize(self)

    # 4. collect statistics
    cluster_centers = cluster_model.cluster_centers_
    cluster_centers *= data_scale

    screen_size = self.screens.shape
    meanscreen  = np.zeros(shape=(n_clusters,screen_size[1],screen_size[2],screen_size[3]))
    cluster_time = np.zeros(shape=(n_clusters,1))
    width = int(np.floor(np.sqrt(n_clusters)))
    length = int(n_clusters/width)
    # f, ax = plt.subplots(length,width)

    for cluster_ind in range(n_clusters):
        indices = (labels==cluster_ind)
        cluster_data = data[indices]
        cluster_time[cluster_ind] = np.mean(self.global_feats['time'][indices])
        meanscreen[cluster_ind,:,:,:] = calc_cluster_im(self,indices)


    # 5. draw cluster indices
    plt.figure(self.fig.number)
    data *= data_scale
    for i in range(n_clusters):
        self.ax_tsne.annotate(i, xy=cluster_centers[i,0:2], size=20, color='r')
    draw_transition_table(transition_table=self.smdp.P, cluster_centers=cluster_centers,
                          meanscreen=meanscreen, tsne=data[:,0:2], color=self.color, black_edges=self.smdp.edges)

    self.cluster_centers =cluster_centers
    self.meanscreen =meanscreen
    self.cluster_time =cluster_time
    plt.show()

def update_slider(self, name, slider):
    def f():
        setattr(self, name, slider.val)
    return f


