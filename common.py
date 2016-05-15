# import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from smdp import SMDP

# def load_hdf5(object_name, path, num_frames=None):
#     print "    Loading " +  object_name
#     obj_file = h5py.File(path + object_name + '.h5', 'r')
#     obj_mat = obj_file['data']
#     return obj_mat[:num_frames]
#
# def save_hdf5(object_name, path, object):
#     print "    Saving " +  object_name
#     with h5py.File(path + object_name +'.h5', 'w') as hf:
#         hf.create_dataset('data', data=object)

def create_trajectory_data(self):
    self.traj_list = []
    for traj_id in xrange(self.hand_craft_feats['n_trajs']-1):
        traj_point_mask = np.asarray(self.hand_craft_feats['traj'])==traj_id
        traj_labels = self.clustering_labels[np.nonzero(traj_point_mask)]
        traj_length = np.sum(traj_point_mask)
        R = np.sum(self.global_feats['reward'][np.nonzero(traj_point_mask)])
        moves = []
        for i in xrange(traj_length-1):
            if traj_labels[i] != traj_labels[i+1] and self.smdp.P[traj_labels[i],traj_labels[i+1]]>0:
                moves.append((traj_labels[i],traj_labels[i+1]))
        self.traj_list.append(
            {
                'R': R,
                'length': traj_length,
                'moves': moves,
                'points': traj_point_mask,
             }
        )

def visualize(self):
    m = self.smdp
    print 'Value mse: %f' % m.score
    m.evaluate_greedy_policy(0)

    # Display 0:
    title = 'White - n_clusters: %d ' % m.n_clusters
    plt.figure(title)
    plt.title(title)
    plt.plot((m.v_dqn-m.v_dqn.mean())/m.v_dqn.std(),'b--',label='DQN')
    plt.plot((m.v_smdp-m.v_smdp.mean())/m.v_smdp.std(),'r-.',label='Semi-MDP')
    plt.plot((m.v_greedy-m.v_greedy.mean())/m.v_greedy.std(),'g',label='improved policy')
    plt.legend()
    plt.xlabel('cluster index')
    plt.ylabel('value')


    # Display 1 (regular)
    # plt.plot(m.v_dqn - m.v_dqn.mean(),'b')
    # plt.plot(m.v_smdp - m.v_smdp.mean(),'r')

    # Display 2 (normalized)
    title = 'Reg - n_clusters: %d ' % m.n_clusters
    plt.figure(title)
    plt.title(title)
    plt.plot(m.v_dqn,'b--',label='DQN')
    plt.plot(m.v_smdp,'r-.',label='Semi-MDP')
    plt.plot(m.v_greedy,'g',label='improved policy')
    plt.legend()
    plt.xlabel('cluster index')
    plt.ylabel('value')

    ll = np.arange(start=0.0,stop=0.05,step=0.05)
    title = 'greedy policy improvement'
    plt.figure(title)
    plt.title(title)
    plt.plot(m.v_smdp,'r--',label='smdp')
    plt.plot(m.v_dqn,'g--',label='dqn')

    for l in ll:
        m.evaluate_greedy_policy(l)
        plt.plot(m.v_greedy,c=np.random.rand(3,1),label=l)
        # plt.plot(m.v_greedy,'b',label='improved policy')
    plt.legend()
    plt.show(block=True)

    plt.figure('Correlation coeficient')
    markerline, stemlines, baseline = plt.stem(self.state_pi_correlation,'b-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)


    # value_diff =  (m.v_greedy[:,0]-m.v_greedy[:,0].mean())/m.v_greedy[:,0].std() - (m.v_smdp-m.v_smdp.mean())/m.v_smdp.std()
    # value_diff =  (m.v_greedy[:,0]-m.v_smdp)/m.v_smdp
    # plt.plot(value_diff,'r--',label='V greedy - V smdp')
    # corr = np.corrcoef(value_diff, self.state_pi_correlation)[0,1]
    plt.xlabel('SMDP state')
    plt.ylabel('Correlation coefient')
    # plt.legend()
    title = 'Correlation between percantage of choosing the greedy policy(at SMDP state i) and the trajectotry total reward'
    plt.title(title)

def reward_policy_correlation(traj_list, policy, smdp):
    N = len(traj_list)
    corr = np.zeros(smdp.n_clusters)

    for c in xrange(smdp.n_clusters):
        rewards = np.zeros(N)
        good_moves = np.zeros(N)
        for t_ind,traj in enumerate(traj_list):
            rewards[t_ind] = traj['R']
            count = 0
            total_cluster_visitations = 0
            for move in traj['moves']:
                if move[0] != c:
                    continue
                total_cluster_visitations += 1
                if policy[int(move[0])][1]==move[1]: # policy[i] = pi_i .
                    count += 1
            if total_cluster_visitations == 0:
                continue
            good_moves[t_ind] = float(count)/total_cluster_visitations

        corr[c] = np.corrcoef(rewards, good_moves)[0,1]

    return corr

def draw_skill_time_dist(self,skill_ind):
    plt.figure('Cluster %d skills' %skill_ind)
    for i in xrange(len(self.smdp.skill_time[skill_ind])):
        subplot_ind = 331+i
        ax = plt.subplot(subplot_ind)
        ax.set_title('Skill %d' %self.smdp.skill_list[skill_ind][i])
        # ax.axis('off')
        plt.hist(self.smdp.skill_time[skill_ind][i], bins=100)


def extermum_trajs_discrepency(traj_list, labels, termination, rewards, values, n_clusters, k=1):

    def unite_trajs_mask(traj_list, traj_indices, n_points):
        unite_mask = np.zeros(n_points, dtype=bool)
        for t_ind in traj_indices:
            unite_mask = np.logical_or(unite_mask, traj_list[t_ind]['points'])
        return unite_mask

    n_trajs = len(traj_list)
    n_points = len(traj_list[0]['points'])
    reward = np.zeros(n_trajs)

    # sort trajectories by reward
    for t_ind, t in enumerate(traj_list):
        reward[t_ind] = t['R']

    traj_order = np.argsort(reward)
    bottom_trajs = traj_order[:k]
    top_trajs = traj_order[-k:]
    top_mask = unite_trajs_mask(traj_list, top_trajs, n_points)
    bottom_mask = unite_trajs_mask(traj_list, bottom_trajs, n_points)

    top_model = SMDP(labels, termination[top_mask], rewards[top_mask], values[top_mask], n_clusters, gamma=0.99, trunc_th = 0.1)
    bottom_model = SMDP(labels, termination[bottom_mask], rewards[bottom_mask], values[bottom_mask], n_clusters, gamma=0.99, trunc_th = 0.1)

    cross_entropy = np.zeros(n_clusters)
    for c in xrange(n_clusters):
        cross_entropy[c] = 0.5 * (scipy.stats.entropy(top_model.P[c], bottom_model.P[c]) + scipy.stats.entropy(bottom_model.P[c], top_model.P[c])) - \
                           scipy.stats.entropy(top_model.P[c]) - scipy.stats.entropy(bottom_model.P[c])

    return cross_entropy

#############################
# 8. color outliers
#############################
# def outliers(event):
#     if self.outlier_color is None:
#         # run your algorithm once
#         from sos import sos
#         import argparse
#         import sys
#         parser = argparse.ArgumentParser(description="Stochastic Outlier Selection")
#         parser.add_argument('-b', '--binding-matrix', action='store_true',
#         default=False, help="Print binding matrix", dest="binding_matrix")
#         parser.add_argument('-t', '--threshold', type=float, default=None,
#         help=("Float between 0.0 and 1.0 to use as threshold for selecting "
#             "outliers. By default, this is not set, causing the outlier "
#             "probabilities instead of the classification to be outputted"))
#         parser.add_argument('-d', '--delimiter', type=str, default=',', help=(
#         "String to use to separate values. By default, this is a comma."))
#         parser.add_argument('-i', '--input', type=argparse.FileType('rb'),
#         default=sys.stdin, help=("File to read data set from. By default, "
#             "this is <stdin>."))
#         parser.add_argument('-m', '--metric', type=str, default='euclidean', help=(
#         "String indicating the metric to use to compute the dissimilarity "
#         "matrix. By default, this is 'euclidean'. Use 'none' if the data set "
#         "is a dissimilarity matrix."))
#         parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
#         default=sys.stdout, help=("File to write the computed outlier "
#             "probabilities to. By default, this is <stdout>."))
#         parser.add_argument('-p', '--perplexity', type=float, default=30.0,
#         help="Float to use as perpexity. By default, this is 30.0.")
#         parser.add_argument('-v', '--verbose', action='store_true', default=False,
#         help="Print debug messages to <stderr>.")
#         args = parser.parse_args()
#         self.outlier_color = sos(self.global_feats['tsne'], 'euclidean', 50,args)
#
#
#
#     self.tsne_scat.set_array(self.outlier_color)
#     sizes = np.ones(self.num_points)*self.pnt_size
#     sizes[self.outlier_color>self.slider_outlier_thresh.val] = 250
#
#     self.tsne_scat.set_sizes(sizes)
#     self.fig.canvas.draw()
#

# self.ax_otlyr = plt.axes([0.80, 0.77, 0.09, 0.02])
# self.b_otlyr = Button(self.ax_otlyr, 'Outliers')
# self.outlier_color = None
# self.b_otlyr.on_clicked(outliers)
# self.slider_outlier_thresh = Slider(plt.axes([0.80, 0.74, 0.09, 0.02]), 'outlier_thresh', valmin=0, valmax=1, valinit=0.75)
# self.SLIDER_FUNCS.append(update_slider(self, 'outlier_thresh', self.slider_outlier_thresh))
# self.slider_outlier_thresh.on_changed(self.update_sliders)
