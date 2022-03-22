import numpy as np
import graph_tool.all as gt
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class NB_model:

    def __init__(self, g):
        self.g = g
        self.vals, self.vecs = self.calc_vertex_embedding()
        self.kmeans_labels, self.k = self.calc_kmeans()
        self.global_cen = None
        self.local_cen = None
        self.local_cen_com = None
        self.centrality = self.build_centrality()

    def calc_vertex_embedding(self):
        g = self.g

        bool_visited = g.new_vp("bool")
        bool_visited.a = g.vp.visited.a.astype(bool)
        g.set_vertex_filter(bool_visited, inverted=True)

        A = gt.adjacency(g)
        D = sp.diags([v.out_degree() for v in g.vertices()])
        Z = sp.csr_matrix((g.num_vertices(), g.num_vertices()))
        I = sp.eye(g.num_vertices(), g.num_vertices())
        up = sp.hstack((A, I - D))
        down = sp.hstack((I, Z))
        B = sp.vstack((up, down))

        if g.num_vertices() <= 1:
            return 1, np.zeros((B.shape))

        counter = 0
        while True:
            try:
                self.vals, self.vecs = sp.linalg.eigs(
                    B, k=min(B.shape[1] / 2, 6), which="LM", tol=0.01, maxiter=5000)
                break
            except:
                counter += 1
                if counter == 5:
                    break

        g.clear_filters()
        return self.vals, self.vecs

    def calc_kmeans(self):
        g = self.g
        vals = self.vals
        vecs = self.vecs
        c = 0.0
        k = 0
        if not np.iscomplex(vals[-1]):
            c = np.real(vals[-1] ** 0.5)
            for val in vals[::-1]:
                if not np.iscomplex(val) and np.real(val) > c:
                    k += 1
                else:
                    break
        else:
            k = 1

        k = min(k, 10)
        emb = np.real(vecs[:g.num_vertices(), -1:-1 - k - 1:-1])
        # community
        kmeans_labels = KMeans(init='k-means++', n_clusters=k,
                                    random_state=42).fit(emb).labels_
        g.vp.kmeans = g.new_vp("int", 0)
        g.vp.kmeans.a = kmeans_labels

        return kmeans_labels, k

    def calc_global_centrality(self):
        g = self.g
        vecs = self.vecs
        bool_visited = g.new_vp("bool")
        bool_visited.a = g.vp.visited.a.astype(bool)
        g.set_vertex_filter(bool_visited, inverted=True)

        global_cen = np.real(vecs[:g.num_vertices(), -1])
        self.global_cen = np.zeros((g.num_vertices(ignore_filter=True), 1))
        vertexMap = [int(v) for v in g.vertices()]
        for ind, val in enumerate(global_cen):
            self.global_cen[vertexMap[ind]] = abs(val)

        g.clear_filters()
        return self.global_cen

    # centrality for each community
    def calc_local_centrality(self):
        g = self.g
        k = self.k

        local_cen = g.new_vp("double", 0.0)
        local_cen_com = [g.new_vp("double", 0.0)
                         for i in range(0, k)]
        local_cen_a = local_cen.a
        local_cen_com_a = [cen.a for cen in local_cen_com]

        for i in range(0, k):
            free_nodes_com_p = g.new_vp("bool")
            free_nodes_com_p.a = np.logical_and(
                np.logical_not(g.vp.visited.a), g.vp.kmeans.a == i)
            g.set_vertex_filter(free_nodes_com_p)
            if g.num_vertices() <= 1:
                continue

            vertexMap = list(g.vertices())
            A = gt.adjacency(g)
            D = sp.diags([v.out_degree() for v in vertexMap])
            Z = sp.csr_matrix((g.num_vertices(), g.num_vertices()))
            I = sp.eye(g.num_vertices(), g.num_vertices())
            up = sp.hstack((A, I - D))
            down = sp.hstack((I, Z))
            B = sp.vstack((up, down))

            counter = 0
            while True:
                try:
                    vals, vecs = sp.linalg.eigs(B, k=1, which="LM", tol=0.005, maxiter=5000)
                    break
                except:
                    counter += 1
                    if counter == 5:
                        break

            for index, cenVal in enumerate(np.real(vecs[:g.num_vertices()])):
                local_cen_a[int(vertexMap[index])] = abs(cenVal)
                local_cen_com_a[i][int(vertexMap[index])] = abs(cenVal)
            g.clear_filters()

        self.local_cen = local_cen_a
        self.local_cen_com = local_cen_com
        return self.local_cen

    def build_centrality(self, is_rebuild=False):
        if is_rebuild:
            self.calc_vertex_embedding()

        global_centrality = self.calc_global_centrality()
        local_centrality = self.calc_local_centrality()
        local_centrality = np.array([[l] for l in local_centrality])
        state = np.hstack((local_centrality, global_centrality, self.kmeans_labels.reshape((self.g.num_vertices(ignore_filter=True), 1))))
        self.g.clear_filters()
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(state)

# g = gt.collection.data["polbooks"]
# g.vp.visited = g.new_vp("bool", False)
# nbm = NB_model(g)
# pos = gt.sfdp_layout(g)
# cen = g.new_vp("double", 0.0)
# cen.a = nbm.global_cen.reshape((105))*30
# print cen.a
# gt.graph_draw(g, pos=pos, vertex_size=cen)

# for v in range(20):
#     g.vp.visited[g.vertex(v)] = True

# g.set_vertex_filter(g.vp.visited, inverted=True)

# nbm.calc_vertex_embedding()
# nbm.build_centrality()
# cen = g.new_vp("double", 0.0)
# cen.a = nbm.global_cen.reshape((105))*30
# g.set_vertex_filter(g.vp.visited, inverted=True)
# print cen.a
# gt.graph_draw(g, pos=pos, vertex_size=cen)
