import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy

class CluSTP:
    def __init__(self, filename):
        # variable need to be defined
        # x[i][j]: whether or not point i connects to point j
        # R[i]: set of points in cluster i
        self.n, self.n_clusters, self.coordinates, self.R, self.source_vertex = self.get_data(filename)
        self.cluster_of_vertices = self.get_cluster_of_vertices(self.R)
        # self.connected_clusters = [set() for i in range(self.n)]
        self.out_vertices_of_cluster = [list() for i in range(self.n_clusters)]

        self.distances = self.calculate_distance()

        # need to update in init_solution and set_value_propagate
        self.x = np.zeros([self.n, self.n])
        # cost[i]: cost of vertice i
        self.cost = np.zeros(self.n)
        # n_out[i] numbers of out-edge of cluster i
        # self.n_out = np.zeros(self.n_clusters)
        self.total_cost = 0

    def get_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        n = int(lines[2][:-1].split(":")[1])  # number of vertices
        n_clusters = int(lines[3][:-1].split(":")[1])

        # coordinates of vertices
        coordinates = []
        for i in range(6, len(lines) - n_clusters - 3):
            numbers = lines[i][:-1].replace("  ", " ").split(' ')
            coordinates.append((int(numbers[1]), int(numbers[2])))

        # R[i]: set of vertices in cluster i
        R = []
        for i in range(len(lines) - n_clusters - 1, len(lines) - 1):
            Ri = set([int(vertex) for vertex in lines[i][:-1].split()[1:-1]])
            R.append(Ri)

        # source vertice
        source_vertex = int(lines[len(lines) - n_clusters - 2][:-1].split(":")[1])

        return n, n_clusters, coordinates, R, source_vertex

    def get_cluster_of_vertices(self, R):
        clusters = np.zeros(self.n)
        for i, Ri in enumerate(R):
            for v in Ri:
                clusters[v] = i

        return clusters

    def calculate_distance(self):
        distances = euclidean_distances(self.coordinates, self.coordinates)
        return distances

    def init_solution(self):
        # 1. tao cay khung cho tung cum
        R = copy.deepcopy(self.R)
        for i, Ri in enumerate(R):
            # print("Cluter", i)
            open_set = Ri
            close_set = set()

            first_vertice = random.sample(open_set, 1)[0]
            open_set.remove(first_vertice)
            close_set.add(first_vertice)

            while len(open_set) != 0:
                random_close_vertice = random.sample(close_set, 1)[0]
                random_open_vertice = random.sample(open_set, 1)[0]
                self.add_edge(random_close_vertice, random_open_vertice)

                open_set.remove(random_open_vertice)
                close_set.add(random_open_vertice)

                # 2. nối các cụm chưa được nối với cụm nào vào các cụm đã nối (cụm đã nối đầu tiên là cụm gốc)
                #         # find root cluster
                #         root_cluster = 0
                #         for i, Ri in enumerate(self.R):
                #             if self.source_vertex in Ri:
                #                 root_cluster = i
                #                 break
                #         print("Root Cluster =", root_cluster)

        # add edges from cluster to cluster
        cluster_indexes = set(range(self.n_clusters))
        open_set = cluster_indexes
        close_set = set()

        first_cluster = random.sample(open_set, 1)[0]
        open_set.remove(first_cluster)
        close_set.add(first_cluster)

        while (len(open_set)) != 0:
            random_close_cluster = random.sample(close_set, 1)[0]
            random_close_vertice = random.sample(self.R[random_close_cluster], 1)[0]

            random_open_cluster = random.sample(open_set, 1)[0]
            random_open_vertice = random.sample(self.R[random_open_cluster], 1)[0]

            self.add_edge(random_close_vertice, random_open_vertice)

#             print('add', random_close_vertice, random_open_vertice)
            open_set.remove(random_open_cluster)
            close_set.add(random_open_cluster)

            # 3. cập nhật cost, total_cost
            #         pass
            self.cost = self.calculate_cost()
            self.total_cost = np.sum(self.cost)

    def add_edge(self, i, j):
        # print('add', i, j)
        self.x[i][j] = 1
        self.x[j][i] = 1

        if self.cluster_of_vertices[i] != self.cluster_of_vertices[j]:
            # self.n_out[int(self.cluster_of_vertices[i])] += 1
            # self.n_out[int(self.cluster_of_vertices[j])] += 1
            #
            # self.connected_clusters[int(self.cluster_of_vertices[i])].add(self.cluster_of_vertices[j])
            # self.connected_clusters[int(self.cluster_of_vertices[j])].add(self.cluster_of_vertices[i])

            self.out_vertices_of_cluster[int(self.cluster_of_vertices[i])].append(i)
            self.out_vertices_of_cluster[int(self.cluster_of_vertices[j])].append(j)

    def remove_edge(self, i, j):
        self.x[i][j] = 0
        self.x[j][i] = 0

        if self.cluster_of_vertices[i] != self.cluster_of_vertices[j]:
            # self.n_out[int(self.cluster_of_vertices[i])] -= 1
            # self.n_out[int(self.cluster_of_vertices[j])] -= 1
            #
            # self.connected_clusters[int(self.cluster_of_vertices[i])].remove(self.cluster_of_vertices[j])
            # self.connected_clusters[int(self.cluster_of_vertices[j])].remove(self.cluster_of_vertices[i])

            self.out_vertices_of_cluster[int(self.cluster_of_vertices[i])].remove(i)
            self.out_vertices_of_cluster[int(self.cluster_of_vertices[j])].remove(j)

    def calculate_cost(self):
        cost = np.zeros(self.n)
        cost[self.source_vertex] = 0
        # Mark all the vertices as not visited
        visited = [False] * self.n

        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        queue.append(self.source_vertex)
        visited[self.source_vertex] = True

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            # print(s, end=" ")

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            neighbors = np.where(self.x[s] == 1)[0]
            for i in neighbors:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                    cost[i] = self.distances[s][i] + cost[s]

        return cost

    def show_graph(self):
        G = nx.Graph()
        edges = np.where(self.x == 1)
        edges = [e for e in zip(list(edges[0]), list(edges[1])) if e[0] <= e[1]]
        G.add_edges_from(edges)
#         print(edges)
#         print(G.nodes())
#         print((G.nodes().items))
#         labels = [str(node) for node in list(G.nodes())]
        labelmap = dict(zip(G.nodes(), list(G.nodes())))
        nx.draw(G, labels=labelmap, with_labels=True)
        plt.show()

    def get_assign_delta(self, out_vertice, connected_vertice, new_out_vertice, new_connected_vertice):
        # return changed value of just one vertice in that cluster
        old_edge_distance = self.distances[out_vertice, connected_vertice]
        new_edge_distance = self.distances[new_out_vertice, new_connected_vertice]

        # n_vertices_in_leaf_cluster = len(self.R[int(self.cluster_of_vertices[out_vertice])])
        return (new_edge_distance - old_edge_distance) #* n_vertices_in_leaf_cluster

    def set_value_propagate(self, out_vertice, connected_vertice, new_out_vertice, new_connected_vertice):
        # update x, out_vertices_of_cluster
        self.add_edge(new_out_vertice, new_connected_vertice)
        self.remove_edge(out_vertice, connected_vertice)

        # update cost, total cost
        vertices_in_leaf_cluster = self.R[int(self.cluster_of_vertices[out_vertice])]
        delta = self.get_assign_delta(out_vertice, connected_vertice, new_out_vertice, new_connected_vertice)
        for vertice in vertices_in_leaf_cluster:
            self.cost[vertice] += delta

        self.total_cost += delta * len(vertices_in_leaf_cluster)

    def get_neighbors(self):
        # chọn cụm bất kỳ có cạnh ra bằng 1: cụm lá
        leaf_clusters = [i for i in range(self.n_clusters) if len(self.out_vertices_of_cluster[i]) == 1] #np.where(self.n_out == 1)[0]
        random_cluster = np.random.choice(leaf_clusters)

        out_vertice = self.out_vertices_of_cluster[random_cluster][0]
        connected_vertice = np.where(self.x[out_vertice, :] == 1)[0][0]
        # current_connected_cluster = self.cluster_of_vertices[connected_vertice]
        # other_clusters = list(range(self.n_clusters)).remove(current_connected_cluster)

        # lấy một đỉnh ở trong random_cluster, nối với những đỉnh bất kỳ khác cụm
        cluster_vertices = self.R[random_cluster]
        other_vertices = set(range(self.n)).difference(cluster_vertices)

        new_combination_vertices = []
        for new_out_vertice in cluster_vertices:
            for new_connected_vertice in other_vertices:
                new_combination_vertices.append((new_out_vertice, new_connected_vertice))

        return (out_vertice, connected_vertice), new_combination_vertices

    def search(self):
        # search solution:
        # choose one leaf cluster, make change inside cluster, move out-egde of cluster to another cluster
        while True:
            old_combination_vertices, new_combination_vertices = self.get_neighbors()
            (out_vertice, connected_vertice) = old_combination_vertices

            deltas = []
            for combination in new_combination_vertices:
                new_out_vertice, new_connected_vertice = combination
                delta = self.get_assign_delta(out_vertice, connected_vertice, new_out_vertice, new_connected_vertice)
                deltas.append(delta)

            min_index = np.argmin(deltas)
            if deltas[int(min_index)] <= 0:
                chosen_combination = new_combination_vertices[int(min_index)]
                self.set_value_propagate(out_vertice, connected_vertice, chosen_combination[0], chosen_combination[1])

            print("Current Cost =", self.total_cost)


    def get_solution_file(self):
        pass

obj = CluSTP('data/Type_1_Small/10berlin52.clt')
obj.init_solution()
# obj.show_graph()
# print(obj.cost)
# print(obj.source_vertex)
obj.search()