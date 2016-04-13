import numpy as np
import scipy.linalg
import itertools

class EMHC(object):
    def __init__ (self, X, termination, min_clusters=20, max_entropy=10, w_entropy=1, w_distance=1e-4):

        # 0. scale features
        # TODO

        # DEBUG
        # N = 150
        # X = X[:N]
        # termination = termination[:N]

        # 2. remove last point (it is loaded with zeros)
        self.X = X[:-1,:]
        self.termination = termination[:-1]
        self.termination[-1] = 1

        # 1. parameters
        self.w_entropy = w_entropy
        self.w_distance = w_distance
        self.n_samples = self.X.shape[0]
        self.n_clusters = self.n_samples
        self.min_clusters = min_clusters
        self.max_entropy = max_entropy

        # 2. total transition matrix TT
        self.init_TT()

        # 3. entropy list e
        self.e = [0]*self.n_samples

        # 4. pairwise distance matrix D
        self.init_D()

        # 5. clusters size vector s
        self.s = [1]*self.n_samples

        # 6. initalize connectivity lists : in_list, out_list
        self.init_connectivity_lists()

        # 7. pairwise entropy gain matrix EG
        self.init_pw_ent_gain_mat()

        # 8. score matrix SCORE
        self.update_score_mat()

        # 9. i_min, j_min
        self.i_min, self.j_min = np.unravel_index(np.argmin(self.SCORE),self.SCORE.shape)

        # 10. assign empty labels
        self.labels_ = -np.ones(self.n_samples)

        # 11. print
        print 'Finished preparing model...'

    # 10. fit function
    def fit(self):
        while self.n_clusters > self.min_clusters and np.sum(self.e) < self.max_entropy:

            # 10.0 decrease number of clusters by 1
            self.n_clusters -= 1

            # 10.1 unite TT i,j rows and columns
            self.TT = self.unite_rows_cols(self.TT, self.i_min, self.j_min)

            # 10.2 update connectivity lists
            self.update_connectivity_lists()

            # 10.3 update vector size
            self.update_cluster_size()

            # 10.4 change list
            self.change_list = [self.i_min] + self.in_list[self.i_min]

            # 10.5 update indices due to removal of j_min
            # removed for the meantime: we operate on the upper triangular part of the pairwise matrix.
            # Therefore i is always smaller than j
            # self.change_list = self.update_indices(self.change_list, self.j_min)

            # 10.6 update entropy
            self.update_entropy()

            # 10.7 update pairwise entropy gain matrix EG
            self.update_pw_ent_gain_mat()

            # 10.8 update pairwise distance matrix D
            self.update_pw_distances_mat()

            # 10.9 update score matrix SCORE
            self.update_score_mat()

            # 10.10 update best pair
            self.i_min, self.j_min = np.unravel_index(np.argmin(self.SCORE), self.SCORE.shape)

            print 'n_clusters: %d' % self.n_clusters

        # 11. assign labels
        for i, cluster in enumerate(self.member_list):
            for c in cluster:
                self.labels_[c] = i

    def add_unique_excluding_i(self, list_a, list_b, i):
        in_a = set(list_a)
        in_b = set(list_b)
        in_b_but_not_in_a = in_b - in_a
        result = list_a + list(in_b_but_not_in_a)
        if i in result: result.remove(i)
        return result

    def dist(self, l, m, L=2):
        u = self.X[l]
        v = self.X[m]
        if L == 1:
            return (u - v).abs().sum()
        if L == 2:
            return ((u - v) ** 2).sum()

    def clusters_dist(self, i, j):
        d = 0
        k = 0
        for pair in itertools.product(self.member_list[i], self.member_list[j]):
            d_pair = self.dist(*pair)
            d += d_pair
            k += 1
        mean_dist = d/k

        return mean_dist

    def build_clusters_vec(d, n_samples):
        clusters_vec = np.zeros(n_samples)
        for cluster_id, cluster in enumerate(d):
            for member in cluster:
                clusters_vec[member] = cluster_id

        return clusters_vec

    def init_TT(self):
        self.TT = np.zeros(shape=(self.n_samples, self.n_samples))
        for i, t in enumerate(self.termination[:-1]):
            if t:
                self.TT[i, i] = 1
            else:
                self.TT[i, i + 1] = 1
        self.TT[-1,-1] = 1

    def init_D(self):
        self.D = np.infty * np.ones(shape=(self.n_samples, self.n_samples))
        for i in xrange(self.n_samples-1):
            for j in xrange(i + 1, self.n_samples):
                self.D[i, j] = self.dist(i, j)

    def init_connectivity_lists(self):
        self.in_list = [[] for x in self.termination]
        self.out_list = [[] for x in self.termination]
        self.member_list = [[x] for x in xrange(self.n_samples)]

        for i, t in enumerate(self.termination):
            if t == 0 and i<(self.n_samples-1):
                self.in_list[i + 1] = [i]
                self.out_list[i] = [i+1]

    def init_pw_ent_gain_mat(self):
        unite_zero_ent = 2*scipy.stats.entropy(np.asarray([0.5,0.5]))
        self.EG = np.zeros(shape=(self.n_samples, self.n_samples))
        self.EG[np.tril_indices(self.n_samples)] = np.infty
        for i in xrange(self.n_samples):
            for j in xrange(i+1,self.n_samples):
                if self.termination[i]==0 and self.termination[j]==0:
                    if abs(i-j)!=1:
                        self.EG[i,j]=unite_zero_ent

    def update_score_mat(self):
        self.SCORE = self.w_entropy*self.EG + self.w_distance*self.D

    def update_connectivity_lists(self):

        # 1. j_min outputs
        # 1.1 in-links
        for c in self.out_list[self.j_min]:
            self.in_list[c].remove(self.j_min) # remove j_min
            if c != self.i_min and (self.i_min not in self.in_list[c]):
                self.in_list[c].append(self.i_min) # add i_min

        # 1.2 out-links
        self.out_list[self.i_min] = self.add_unique_excluding_i(self.out_list[self.i_min], self.out_list[self.j_min], self.i_min)
        self.out_list[self.j_min] = []

        # 2. j_min inputs
        # 2.1 out-links
        for c in self.in_list[self.j_min]:
            self.out_list[c].remove(self.j_min)  # remove j_min
            if c != self.i_min and (self.i_min not in self.out_list[c]):
                self.out_list[c].append(self.i_min)  # add i_min

        # 2.2 in-links
        self.in_list[self.i_min] = self.add_unique_excluding_i(self.in_list[self.i_min], self.in_list[self.j_min], self.i_min)
        self.in_list[self.j_min] = []

        # 3. decrease cluster numbers larger than j_min
        # 3.1 in list
        for ii, cluster in enumerate(self.in_list):
            for jj, c in enumerate(cluster):
                if c > self.j_min:
                    self.in_list[ii][jj] -= 1

        # 3.2 out list
        for ii, cluster in enumerate(self.out_list):
            for jj, c in enumerate(cluster):
                if c > self.j_min:
                    self.out_list[ii][jj] -= 1

        # 4. remove j_min from connectivity lists
        del self.out_list[self.j_min]
        del self.in_list[self.j_min]

        # 5. member list
        self.member_list[self.i_min] += self.member_list[self.j_min]
        del self.member_list[self.j_min]

    def unite_rows_cols(self, A, i, j):
        B = np.copy(A)
        B[i] += B[j]
        B[:,i] += B[:,j]
        B = np.delete(B,j,axis=0)
        B = np.delete(B,j,axis=1)
        return B

    def update_cluster_size(self):
        self.s[self.i_min] += self.s[self.j_min]
        self.s = np.delete(self.s,self.j_min)

    def update_indices(self, arry, j):
        arry_shift = arry
        for a in arry_shift:
            if a>j:
                a -= 1
        return arry_shift

    def update_entropy(self):
        for i in self.change_list:
            self.e[i] = scipy.stats.entropy(self.TT[i]/self.s[i])
        self.e = np.delete(self.e,self.j_min)

    def update_pw_ent_gain_mat(self):
        self.EG = np.delete(self.EG, self.j_min, axis=0)
        self.EG = np.delete(self.EG, self.j_min, axis=1)

        for i in self.change_list:
            for j in xrange(self.n_clusters):
                if i==j:
                    continue

                # 1. create TT matrix based on i,j union
                TTij = self.unite_rows_cols(self.TT, i, j)

                # 1.5 index change: if i>j then i should be decreased by 1
                if i>j:
                    i -= 1
                    in_list_i = list(self.in_list[i])
                    in_list_j = list(self.in_list[j])
                    for ii in xrange(len(in_list_i)):
                        if in_list_i[ii]>j:
                            in_list_i[ii] -= 1

                for jj in xrange(len(in_list_j)):
                    if in_list_j[jj] > j:
                        in_list_j[jj] -= 1

                # 2. entropy gain
                d_ent = 0

                # 2.1 add entropy of the newly created cluster
                d_ent += scipy.stats.entropy(TTij[i]/(self.s[i]+self.s[j]))

                # 2.2 remove old entropy of i,j
                d_ent -= self.s[i] * self.e[i]
                d_ent -= self.s[j] * self.e[j]

                # 2.3 remove old entropy of all clusters pointing to i,j
                d_ent -= np.sum(self.s[self.in_list[i]] * self.e[self.in_list[i]])
                d_ent -= np.sum(self.s[self.in_list[j]] * self.e[self.in_list[j]])

                # 2.4 add new entropy of all clusters pointing to i,j
                for l in (in_list_i + in_list_j):
                    TTij_l = TTij[l]
                    d_ent += scipy.stats.entropy(TTij_l/TTij_l.sum())

                if j>i:
                    self.EG[i,j] = d_ent
                else:
                    self.EG[j,i] = d_ent

    def update_pw_distances_mat(self):
        self.D = np.delete(self.D, self.j_min, axis=0)
        self.D = np.delete(self.D, self.j_min, axis=1)

        for i in self.change_list:
            for j in xrange(self.n_clusters):
                if i == j:
                    continue

                d_pair = self.clusters_dist(i,j)
                if j>i:
                    self.D[j,i] = d_pair
                else:
                    self.D[i,j] = d_pair