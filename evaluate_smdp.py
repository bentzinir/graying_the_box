import numpy as np
import scipy.linalg

def divide_tt(X, tt_ratio):
    N = X.shape[0]
    X_train = X[:int(tt_ratio*N)]
    X_test = X[int(tt_ratio*N):]
    return X_train, X_test

def calculate_transition_matrix(termination, labels, n_clusters):
    TT = np.zeros((n_clusters, n_clusters))
    for i, (t, l) in enumerate(zip(termination[:-1], labels[:-1])):
        if t:
            TT[labels[i], labels[i]] += 1
        else:
            TT[labels[i], labels[i + 1]] += 1
    T = TT / TT.sum(axis=1)[:, np.newaxis]
    return T,TT

def entropy_inertia(termination, labels, n_clusters):
    T_mat, TT_mat = calculate_transition_matrix(termination, labels, n_clusters)
    entropy = scipy.stats.entropy(T_mat.T)
    entropy[np.nonzero(np.isnan(entropy))] = 0
    entropy[np.nonzero(np.isinf(entropy))] = 0
    weighted_entropy = np.average(a=entropy, weights=TT_mat.sum(axis=1))
    return weighted_entropy

def reward_coherency(rewards, labels, n_clusters, tt_ratio):
    # 0. divide data into train/test
    rewards_train, rewards_test = divide_tt(rewards, tt_ratio)
    labels_train, labels_test = divide_tt(labels, tt_ratio)

    # 1. evaluate train reward statistics
    train_vec = np.zeros(shape=(n_clusters,2))
    for i,(l,r) in enumerate(zip(labels_train,rewards_train)):
        train_vec[l, 0] += r
        train_vec[l, 1] += 1

    # 1.5 normalize rewards
    for t in train_vec:
        t[0] = t[0]/t[1]

    # 2. variance of test rewards with respect to the average train reward
    total_var = 0
    for i, (l, r) in enumerate(zip(labels_test, rewards_test)):
        total_var += (r-train_vec[l,0])**2

    likelihood = - total_var / labels_test.shape[0]

    return likelihood

def traj_coherency(rewards, labels, n_clusters, tt_ratio, gamma):
    # 0. divide data into train/test
    rewards_train, rewards_test = divide_tt(rewards, tt_ratio)
    labels_train, labels_test = divide_tt(labels, tt_ratio)

    # 1. helper function
    def avg_traj_reward_estimation(rewards, labels, test_mode, stats_vec=None):
        l_p = labels[0]
        total_r = rewards[0]
        t = 0
        total_var = 0
        k = 0

        if not test_mode:
            stats_vec = np.zeros(shape=(n_clusters, 2))

        for i, (l, r) in enumerate(zip(labels[1:], rewards[1:])):
            if l == l_p:
                total_r += gamma**t * r
                t += 1
            else:
                if test_mode:
                    total_var += (total_r - stats_vec[l_p,0])**2
                    k += 1
                else:
                    stats_vec[l_p,0] += total_r
                    stats_vec[l_p,1] += 1
                    l_p = l
                    total_r = r
                    t = 1

        for t in stats_vec:
            t[0] = t[0] / t[1]

        if test_mode:
            return total_var / k
        else:
            return stats_vec

    # 2. evaluate train trajectory reward
    mean_vec = avg_traj_reward_estimation(rewards_train, labels_train, test_mode=0)

    # 2. evaluate test
    reward_traj_var = avg_traj_reward_estimation(rewards_test, labels_test, test_mode=1, stats_vec=mean_vec)

    likelihood = - reward_traj_var

    return likelihood

def transition_coherency(labels, termination, n_clusters, tt_ratio):
    # 0. divide data into train/test
    term_train, term_test = divide_tt(termination, tt_ratio)
    labels_train, labels_test = divide_tt(labels, tt_ratio)

    # 1. create train TT matrix
    T_train, _ = calculate_transition_matrix(term_train, labels_train, n_clusters)

    # 2. play unseen tranjectory
    likelihood = 0
    k = 1
    for i in labels_test[:-1]:
        if labels_test[i] != labels_test[i+1]:
            likelihood += np.log(T_train[labels_test[i],labels_test[i+1]])
            k += 1

    likelihood = likelihood / k

    return likelihood

def evaluate_smdp(method, rewards, labels, termination, n_clusters):
    tt_ratio = 0.8
    gamma = 0.99
    likelihood_measures = []
    # likelihood_measures.append(entropy_inertia(termination, labels, n_clusters))
    likelihood_measures.append(reward_coherency(rewards, labels, n_clusters, tt_ratio))
    likelihood_measures.append(traj_coherency(rewards, labels, n_clusters, tt_ratio, gamma))
    likelihood_measures.append(transition_coherency(labels, termination, n_clusters, tt_ratio))

    return likelihood_measures

