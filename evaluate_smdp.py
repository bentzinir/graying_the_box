import numpy as np
import scipy.linalg


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
    entropy[np.isnan(entropy) or np.isinf(entropy)] = 0
    weighted_entropy = np.average(a=entropy, weights=TT_mat.sum(axis=1))
    print 'Entropy/Inertia score: %f' % weighted_entropy

def reward_coherency(rewards_train, labels_train, rewards_test, labels_test, n_clusters):

    # 1. evaluate train reward statistics
    train_vec = np.zeros(n_clusters,2)
    for i,(l,r) in enumerate(zip(labels_train,rewards_train)):
        train_vec[l, 0] += r
        train_vec[l, 1] += 1

    # 1.5 normalize rewards
    for t in train_vec:
        t[0] = t[0]/t[1]

    # 2. evaluate test reward statistics
    test_vec = np.zeros(n_clusters, 2)
    for i, (l, r) in enumerate(zip(labels_test, rewards_test)):
        test_vec[l, 0] += r
        test_vec[l, 1] += 1

    # 2.5 normalize rewards
    for t in test_vec:
        t[0] = t[0] / t[1]

    # 3. score: square differences between train/test estimations of reward. weighted by test sample size
    score = np.average(a=(train_vec[:,0]-test_vec[:,0])**2,weights=test_vec[:,1])

    print 'In-state reward score: %f' % score

def traj_coherency(rewards_train, labels_train, rewards_test, labels_test, n_clusters):

    def get_traj_reward_estimation(rewards, labels):
        vec = np.zeros(n_clusters, 2)
        l_p = labels_train[0]
        total_r = rewards_train[0]
        for i, (l, r) in enumerate(zip(labels[1:], rewards[1:])):
            if l != l_p:
                vec[l_p,0] += total_r
                vec[l_p, 1] += 1
                l_p = l
                total_r = r
            else:
                total_r += r

        for t in vec:
            t[0] = t[0] / t[1]

        return vec

    # 1. evaluate train
    train_vec = get_traj_reward_estimation(rewards_train, labels_train)

    # 2. evaluate test
    test_vec = get_traj_reward_estimation(rewards_test, labels_test)

    # 3. score: square difference between train/test estimations of trajectory rewards. weighted by test sample size
    score = np.average(a=(train_vec[:, 0] - test_vec[:, 0]) ** 2, weights=test_vec[:, 1])

    print 'In-state trajectory reward score: %f' % score

def transition_coherency(labels_train, termination_train, labels_test, termination_test, n_clusters):
    # 1. create train TT matrix
    T_train, _ = calculate_transition_matrix(termination_train, labels_train, n_clusters)

    # 2. create test TT matrix
    T_test, _ = calculate_transition_matrix(termination_test, labels_test, n_clusters)

    # 3. calculate KL divergence between each state in train and test
    KL_vec = np.zeros(n_clusters)
    for i, (t_train,t_test) in enumerate(zip(T_train,T_test)):
        KL_vec[i] = scipy.stats.entropy(pk=t_test,qk=t_train)

    # 4. score: square weighted average of KL divergence values weighted by test cluster sizes
    score = np.average(a=KL_vec, weights=T_test.sum(axis=1))

    print 'Transition coherency score: %f' % score

def evaluate_smdp(method, rewards_train, labels_train, termination_train, rewards_test, labels_test, termination_test, n_clusters):
    switcher = {
        0: entropy_inertia(termination_test, labels_test),
        1: reward_coherency(rewards_train, labels_train, rewards_test, labels_test, n_clusters),
        2: traj_coherency(labels_train, termination_train, labels_test, termination_test, n_clusters),
        3: transition_coherency(labels_train, termination_train, labels_test, termination_test, n_clusters),
    }

    return switcher.get(method)

