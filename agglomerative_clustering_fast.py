import numpy as np
import scipy.linalg
import itertools

def dist(X, ind_u, ind_v, L=2):
    u = X[ind_u]
    v = X[ind_v]

    if L==1:
        return (u-v).abs().sum()
    if L==2:
        return ((u-v)**2).sum()

def build_transition_mat(n_clusters, connectivity_vec, clusters_vec):
    T = np.zeros(shape=(n_clusters,n_clusters))

    for i,j in enumerate(connectivity_vec):
        T[clusters_vec[i],clusters_vec[j]] += 1

    return T

def build_clusters_vec(d, n_samples):
    clusters_vec = np.zeros(n_samples)
    for cluster_id, cluster in enumerate(d):
        for member in cluster:
            clusters_vec[member] = cluster_id

    return clusters_vec

def init_TT(termination, n_samples):
    TT = np.zeros(shape=(n_samples,n_samples))
    for i,t in enumerate(termination):
        if t:
            TT[i,i+1] = i
        else:
            TT[i,i] = i
    return TT

def init_D(X, n_samples):
    D = np.infty * np.ones(shape=(n_samples,n_samples))
    for i in n_samples:
        for j in xrange(i+1,n_samples):
            D[i,j] = dist(X,i,j)
    return D

def init_connectivity_lists(termination):

    in_list = [[] for x in termination]

    for i,t in enumerate(termination[:-1]):
        if t==0:
            in_list[i+1] = i

def init_pw_ent_mat(in_list, e, n_samples):
    E = np.infty * np.ones(shape=(n_samples,n_samples))


def agg_clustering(X, termination, min_clusters=20, max_entropy=1):

    # 0. scale features
    # TODO

    # 2. remove last point (it is loaded with zeros)
    X = X[:-1,:]
    termination = termination[:,-1]

    # 1. parameters
    w_entropy = 1
    w_distance = 1e-4
    n_samples = X.shape[0]

    # 2. inititializations
    # 2.1. total transition matrix TT
    TT = init_TT(termination, n_samples)

    # 2.2. entropy list e
    e = [0]*n_samples

    # 2.3. pairwise distance matrix D
    D = init_D(X,n_samples)

    # 2.4. clusters size vector s
    s = [1]*n_samples

    # 2.5.

    n_samples = X.shape[0]
    n_clusters = n_samples
    clusters_vec = np.asarray([x for x in xrange(n_clusters)]) # a list where each element (cluster) holds the indices of its members
    d = [[x] for x in xrange(n_samples)]

    # DEBUG: check performance when starting with k<<n_samples clusters
    n_clusters = 100
    clusters_vec = np.random.randint(low=0,high=n_clusters,size=n_samples)
    d = [[] for x in range(n_clusters)]
    for i,c in enumerate(clusters_vec):
        d[c].append(i)

    # 2. connectivity vector (mapping points to next-points along the trajectory)
    connectivity_vec = -1*np.ones(n_samples)
    for i in xrange(n_samples):
        if termination[i]:
            connectivity_vec[i] = i
        else:
            connectivity_vec[i] = i+1
    connectivity_vec[-1] = n_samples-1

    # 3. Build transition matrix from clusters_vec
    T = build_transition_mat(n_clusters, connectivity_vec, clusters_vec)
    entropy = scipy.stats.entropy(T.T).mean()

    # 4. loop
    while n_clusters > min_clusters and entropy < max_entropy:

        D = np.infty*np.ones(shape=(n_clusters,n_clusters))
        E = np.infty*np.ones(shape=(n_clusters,n_clusters))

        for i in xrange(n_clusters):
            for j in xrange(i+1,n_clusters):

                # 4.1 compute pairwise distances
                for pair in itertools.product(d[i],d[j]):
                    d_pair = dist(X, *pair)
                    if d_pair < D[i,j]:
                        D[i,j] = d_pair

                # 4.2 compute i,j entropy gain
                d_ij = list(d)
                d_ij[i] = d[i]+d[j]
                del d_ij[j]
                clusters_vec_ij = build_clusters_vec(d_ij, n_samples)
                Tij = build_transition_mat(n_clusters-1, connectivity_vec, clusters_vec_ij)
                np.fill_diagonal(Tij,0)
                Tij = Tij[~np.all(Tij==0, axis=1)]
                Tij=Tij/Tij.sum(axis=1)[:,np.newaxis]
                E[i,j] = scipy.stats.entropy(Tij.T).mean() weighted average

        # 4.3 unite pair with lowest score
        score = w_entropy * E + w_distance * D
        ii,jj = np.unravel_index(score.argmin(), score.shape)

        # 4.4 unite clusters i and j
        d[ii] = d[ii]+d[jj]
        del d[jj]

        # 4.5 decrease clusters number by 1
        n_clusters -= 1

        print 'n_clusters: %d' % n_clusters

    return d