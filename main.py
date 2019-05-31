import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
import heapq as priority_queue
from collections import defaultdict
import utils


class CluSTP:

    def __init__(self, filename, graph_type):
        # variable need to be defined
        # x[i][j]: whether or not point i connects to point j
        # R[i]: set of points in cluster i
        self.time_cost = 0
        if graph_type == "Euclid":
            self.n, self.n_clusters, self.coordinates, self.R, self.source_vertex = utils.get_data_euclid(
                filename)
            self.distances = self.calculate_distance()
        elif graph_type == "Non_Euclid":
            self.n, self.n_clusters, self.distances, self.R, self.source_vertex = utils.get_data_non_euclid(
                filename)
        self.cluster_of_vertices = self.get_cluster_of_vertices(self.R)
        # self.connected_clusters = [set() for i in range(self.n)]
        self.out_vertices_of_cluster = [list() for i in range(self.n_clusters)]

        # need to update in init_solution and set_value_propagate
        self.x = np.zeros([self.n, self.n])
        # cost[i]: cost of vertex i
        self.cost = np.zeros(self.n)
        # n_out[i] numbers of out-edge of cluster i
        # self.n_out = np.zeros(self.n_clusters)
        self.total_cost = 0
        self.dijkstra_cost = np.zeros(self.n)
        self.dijkstra_result = [0]*self.n
        self.dijkstra_cost = self.calculate_dijkstra_cost()

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

            first_vertex = random.sample(open_set, 1)[0]
            open_set.remove(first_vertex)
            close_set.add(first_vertex)

            while len(open_set) != 0:
                random_close_vertex = random.sample(close_set, 1)[0]
                random_open_vertex = random.sample(open_set, 1)[0]
                self.add_edge(random_close_vertex, random_open_vertex)

                open_set.remove(random_open_vertex)
                close_set.add(random_open_vertex)

        # 2. nối các cụm chưa được nối với cụm nào vào các cụm đã nối
        # add edges from cluster to cluster
        cluster_indexes = set(range(self.n_clusters))
        open_set = cluster_indexes
        close_set = set()

        first_cluster = random.sample(open_set, 1)[0]
        open_set.remove(first_cluster)
        close_set.add(first_cluster)

        while (len(open_set)) != 0:
            random_close_cluster = random.sample(close_set, 1)[0]
            random_close_vertex = random.sample(
                self.R[random_close_cluster], 1)[0]

            random_open_cluster = random.sample(open_set, 1)[0]
            random_open_vertex = random.sample(
                self.R[random_open_cluster], 1)[0]

            self.add_edge(random_close_vertex, random_open_vertex)

#             print('add', random_close_vertex, random_open_vertex)
            open_set.remove(random_open_cluster)
            close_set.add(random_open_cluster)

            # 3. cập nhật cost, total_cost
            #         pass
            self.cost = self.calculate_cost(self.x, self.source_vertex)
            self.total_cost = np.sum(self.cost)
            # print("Connect cluster", random_close_cluster, random_open_cluster)

        # Dijkstra for source vertex
        list_vertex = list(self.R[int(self.cluster_of_vertices[self.source_vertex])])
        num_vertex = len(list_vertex)
        for i in range(num_vertex):
            for j in range(num_vertex):
                if self.x[list_vertex[i]][list_vertex[j]] == 1:
                    self.remove_edge(list_vertex[i], list_vertex[j])

        self.x += self.dijkstra_result[self.source_vertex]

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

            self.out_vertices_of_cluster[
                int(self.cluster_of_vertices[i])].append(i)
            self.out_vertices_of_cluster[
                int(self.cluster_of_vertices[j])].append(j)

    def remove_edge(self, i, j):
        self.x[i][j] = 0
        self.x[j][i] = 0

        if self.cluster_of_vertices[i] != self.cluster_of_vertices[j]:
            # self.n_out[int(self.cluster_of_vertices[i])] -= 1
            # self.n_out[int(self.cluster_of_vertices[j])] -= 1
            #
            # self.connected_clusters[int(self.cluster_of_vertices[i])].remove(self.cluster_of_vertices[j])
            # self.connected_clusters[int(self.cluster_of_vertices[j])].remove(self.cluster_of_vertices[i])

            self.out_vertices_of_cluster[
                int(self.cluster_of_vertices[i])].remove(i)
            self.out_vertices_of_cluster[
                int(self.cluster_of_vertices[j])].remove(j)


    def calculate_cost(self, x, source_vertex):
        self.time_cost += 1
        list_vertex = self.R[int(self.cluster_of_vertices[source_vertex])]
        cost = np.zeros(self.n)
        cost[source_vertex] = 0
        # Mark all the vertices as not visited
        visited = [False] * self.n

        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        queue.append(source_vertex)
        visited[source_vertex] = True

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            # print(s, end=" ")

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            neighbors = np.where(x[s] == 1)[0]
            if source_vertex != self.source_vertex:
                neighbors = list(
                    set(np.where(x[s] == 1)[0]).intersection(list_vertex))
            for i in neighbors:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                    cost[i] = self.distances[s][i] + cost[s]
        return cost

    def show_graph(self):
        G = nx.Graph()
        edges = np.where(self.x == 1)
        edges = [e for e in zip(list(edges[0]), list(edges[1])) if e[
            0] <= e[1]]
        G.add_edges_from(edges)
#         print(edges)
#         print(G.nodes())
#         print((G.nodes().items))
#         labels = [str(node) for node in list(G.nodes())]
        labelmap = dict(zip(G.nodes(), list(G.nodes())))
        nx.draw(G, labels=labelmap, with_labels=True)
        plt.show()

    def get_assign_delta(self, out_vertex, connected_vertex, new_out_vertex, new_connected_vertex):
        # return changed value of just one vertex in that cluster
        old_edge_distance = self.distances[out_vertex, connected_vertex]
        new_edge_distance = self.distances[
            new_out_vertex, new_connected_vertex]

        # n_vertices_in_leaf_cluster = len(self.R[int(self.cluster_of_vertices[out_vertex])])
        # * n_vertices_in_leaf_cluster
        return (new_edge_distance - old_edge_distance) + (self.cost[new_connected_vertex] - self.cost[connected_vertex])

    def set_value_propagate(self, out_vertex, connected_vertex, new_out_vertex, new_connected_vertex):
        # update x, out_vertices_of_cluster
        self.remove_edge(out_vertex, connected_vertex)
        self.add_edge(new_out_vertex, new_connected_vertex)

        # update cost, total cost
        # vertices_in_leaf_cluster = self.R[
        #     int(self.cluster_of_vertices[out_vertex])]
        # delta = self.get_assign_delta(
        #     out_vertex, connected_vertex, new_out_vertex, new_connected_vertex)
        # for vertex in vertices_in_leaf_cluster:
        #     self.cost[vertex] += delta

        list_vertex = list(self.R[int(self.cluster_of_vertices[out_vertex])])
        num_vertex = len(list_vertex)
        for i in range(num_vertex):
            for j in range(num_vertex):
                if self.x[list_vertex[i]][list_vertex[j]] == 1:
                    self.remove_edge(list_vertex[i], list_vertex[j])

        self.x += self.dijkstra_result[new_out_vertex]

        # self.total_cost += delta * len(vertices_in_leaf_cluster)

    def get_neighbors(self):
        # chọn cụm bất kỳ có cạnh ra bằng 1: cụm lá // hiện tại không lại cụm chứa đỉnh gốc vì có khó
        # khăn ở bước thay đổi cost
        leaf_clusters = [i for i in range(self.n_clusters) if len(
            self.out_vertices_of_cluster[i]) == 1]  # np.where(self.n_out == 1)[0]
        if self.cluster_of_vertices[self.source_vertex] in leaf_clusters:
            leaf_clusters.remove(self.cluster_of_vertices[self.source_vertex])
        random_cluster = np.random.choice(leaf_clusters)

        out_vertices = self.out_vertices_of_cluster[random_cluster]
        # TODO
        # get all out vertex, not random
        # out_vertex = random.sample(out_vertices, 1)[0]
        old_combination_vertices = []
        for out_vertex in out_vertices:
            connected_vertices = np.where(self.x[out_vertex, :] == 1)[0]
            # must choose vertex that are not in the same cluster
            connected_vertices = set(connected_vertices).difference(
                self.R[random_cluster])
            connected_vertex = random.sample(connected_vertices, 1)[0]
            # current_connected_cluster = self.cluster_of_vertices[connected_vertex]
            # other_clusters = list(range(self.n_clusters)).remove(current_connected_cluster)
            old_combination_vertices.append((out_vertex, connected_vertex))

        # lấy một đỉnh ở trong random_cluster, nối với những đỉnh bất kỳ khác cụm
        cluster_vertices = self.R[random_cluster]
        other_vertices = set(range(self.n)).difference(cluster_vertices)
        # TODO
        # get random 10% vertices, it could increase cost
        # you can prevent it by cancelling set_value_propagate when get higher cost
        # other_vertices = random.sample(other_vertices, int(0.1 * len(other_vertices)))

        new_combination_vertices = []
        for new_out_vertex in cluster_vertices:
            for new_connected_vertex in other_vertices:
                # if self.cluster_of_vertices[new_out_vertex] == self.cluster_of_vertices[new_connected_vertex]:
                    # print(1)
                new_combination_vertices.append(
                    (new_out_vertex, new_connected_vertex))

        return old_combination_vertices, new_combination_vertices

    # INSIDE CLUSTER
    def remove_all_edge_in_cluster(self, cluster):
        pass

    def dijkstra_cluster(self, source_vertex):
        new_edge = {}
        S1 = defaultdict(lambda: False)
        p_queue = []
        min_distance = defaultdict(lambda: float("inf"))
        list_vertex = list(
            self.R[int(self.cluster_of_vertices[source_vertex])])
        num_vertex = len(list_vertex)
        min_distance[source_vertex] = 0
        priority_queue.heappush(p_queue, (0, source_vertex))
        current_num_vertex = 0
        while(current_num_vertex < num_vertex):
            for u in p_queue:
                if S1[u[1]] == False:
                    break
            S1[u[1]] = True
            current_num_vertex += 1
            for v in list_vertex:
                uv = self.distances[u[1]][v]
                if min_distance[v] > min_distance[u[1]] + uv:
                    min_distance[v] = min_distance[u[1]] + uv
                    priority_queue.heappush(p_queue, (min_distance[v], v))
                    new_edge[v] = u[1]

        return new_edge
        # Sau khi tim duoc cac canh moi o new_edge, su dung no de cap nhat lai cay
        # khung trong cum dang xet

        # Remove all edge:
        # for i in range(num_vertex):
        # for j in range(num_vertex):
        # if self.x[list_vertex[i]][list_vertex[j]] == 1:
        # self.remove_edge(list_vertex[i], list_vertex[j])
        # Update edge new cost

    def calculate_dijkstra_cost(self):
        dijkstra_cost = np.zeros(self.n)
        for i in range(self.n):
            new_edge = self.dijkstra_cluster(i)
            # size = len(self.R[int(self.cluster_of_vertices[i])])
            temp_x = np.zeros([self.n, self.n])
            for j in new_edge.keys():
                temp_x[j][new_edge[j]] = 1
                temp_x[new_edge[j]][j] = 1
            cost = self.calculate_cost(temp_x, i)
            dijkstra_cost[i] = np.sum(cost)
            self.dijkstra_result[i] = temp_x
        return dijkstra_cost

    def get_assign_delta_inside_cluster(self, cluster, new_edge, old_edge):
        pass

    def set_value_propagate_inside_cluster(self, cluster, new_edge, old_edge):
        pass

    def find_cycle_inside_cluster(self, source):

        # Use DFS
        pass

    # A recursive function that uses visited[] and parent to detect
    # cycle in subgraph reachable from vertex v.
    def isCyclicUtil(self, v, visited, parent, parents):

        # Mark the current node as visited
        visited[v] = True
        parents[v] = parent

        # Recur for all the vertices adjacent to this vertex
        cluster_vertices = self.R[self.cluster_of_vertices[v]]
        neighbors = np.where(self.x[v] == 1)[0]
        neighbors = set(neighbors).intersection(set(cluster_vertices))

        for i in neighbors:
            # If the node is not visited then recurse on it
            if visited[i] == False:
                if (self.isCyclicUtil(i, visited, v)):
                    return True, parents
            # If an adjacent vertex is visited and not parent of current vertex,
            # then there is a cycle
            elif parent != i:
                return True, parents

        return False, parents

    # Returns true if the graph contains a cycle, else false.
    def isCyclic(self):
        parents = np.zeros(self.n)
        # Mark all the vertices as not visited
        visited = [False] * (self.n)
        # Call the recursive helper function to detect cycle in different
        # DFS trees
        for i in range(self.n):
            if visited[i] == False:  # Don't recur for u if it is already visited
                if (self.isCyclicUtil(i, visited, -1, parents)) == True:
                    return True, parents

        return False, parents

    def get_cycle(self):
        is_having_cycle, parents = self.isCyclic()
        if is_having_cycle:
            pass
        else:
            print("PLEASE CHECK, THERE ARE NO CYCLE")

    def get_neighbors_inside_cluster(self):
        random_cluster = np.random.randint(self.n_clusters)
        while len(self.R[random_cluster]) <= 2:  # không xét đến cluster có 2 đỉnh
            random_cluster = np.random.randint(self.n_clusters)
        # tạo chu trình trong cluster đc chọn này, sau đó loại bỏ 1 cạnh trong
        # chu trình để được cây khung mới
        cluster_vertices = self.R[random_cluster]

        # use DFS

    # SEARCH

    def search(self):
        # search solution:
        # choose one leaf cluster, make change inside cluster, move out-egde of
        # cluster to another cluster
        # print("Init out vertices of cluster =", self.out_vertices_of_cluster)
        it = 1
        best = self.total_cost

        while True and it < 100:
            old_combination_vertices, new_combination_vertices = self.get_neighbors()
            # TODO
            # old_combination_vertices is a list
            deltas = []
            pair_of_combinations = []

            for old_combination in old_combination_vertices:
                (out_vertex, connected_vertex) = old_combination
                for new_combination in new_combination_vertices:
                    new_out_vertex, new_connected_vertex = new_combination
                    delta = self.get_assign_delta(
                        out_vertex, connected_vertex, new_out_vertex, new_connected_vertex)
                    delta *= len(self.R[int(self.cluster_of_vertices[out_vertex])])
                    delta += self.dijkstra_cost[new_out_vertex] - self.dijkstra_cost[out_vertex]
                    deltas.append(delta)
                    pair_of_combinations.append((old_combination, new_combination))

            min_index = np.argmin(deltas)

            # CHOOSE TWO COMBINATIONS
            old_chosen_combination, new_chosen_combination = pair_of_combinations[min_index]
             # = new_combination_vertices[int(min_index)]
            out_vertex, connected_vertex = old_chosen_combination
            self.set_value_propagate(out_vertex, connected_vertex, new_chosen_combination[
                                     0], new_chosen_combination[1])
            # self.dijkstra_cluster(out_vertex)
            self.cost = self.calculate_cost(self.x, self.source_vertex)
            self.total_cost = np.sum(self.cost)
            # print("Remove", out_vertex, connected_vertex,
            #       "from cluster", self.cluster_of_vertices[out_vertex],
            #       "to cluster", self.cluster_of_vertices[connected_vertex],
            #       "\nAdd", chosen_combination[0], chosen_combination[1],
            #       "from cluster", self.cluster_of_vertices[chosen_combination[0]],
            #       "to cluster", self.cluster_of_vertices[chosen_combination[1]],)
            # print("Step", it, "\tDelta", deltas[min_index], "\tCurrent Cost =", self.total_cost,
            #       "\t", "out vertices =", self.out_vertices_of_cluster, "\ttime cost", self.time_cost)
            it += 1
            # print("--------------------------------------------------------------")
            if best > self.total_cost:
                # self.show_graph()
                best = self.total_cost

    def get_solution_file(self):
        pass

    def load_result(self, filename):
        X = utils.get_result(filename)
        if X.shape[0] != X.shape[1] or X.shape[0] != self.n:
            print("Error solution")
            return
        self.x.fill(0)
        for i in range(self.n):
            for j in range(self.n):
                if X[i][j] == 1 and self.x[i][j] != 1:
                    self.add_edge(i, j)
        self.cost = self.calculate_cost(self.x, self.source_vertex)
        self.total_cost = np.sum(self.cost)


# obj = CluSTP(filename='data/Euclid/Type_1_Small/5berlin52.clt',
#              graph_type="Euclid")
# obj.init_solution()
# print("Total cost init: " + str(obj.total_cost))
# sol = 'GAsol.opt'
# obj.load_result(
#     'data/Result/Type_1_Small/Para_File(GA_Clus_Tree_5berlin52)_Instance(5berlin52)/LocalSearch/' + sol)

# obj = CluSTP(filename='data/Non_Euclid/Type_1_Small/5berlin52.clt', graph_type="Non_Euclid")
# print("Total cost init: " + str(obj.total_cost))
# sol = 'Para_File(GA_Clus_Tree_5berlin52)_Instance(5berlin52)_Seed(19).opt'
# # obj.load_result('data/Result/Type_1_Small/Para_File(GA_Clus_Tree_5berlin52)_Instance(5berlin52)/LocalSearch/' + sol)
#
# # obj.init_solution()
# print("Total cost before local seach: " + str(obj.total_cost))
# # obj.show_graph()
# # print(obj.source_vertex)
# obj.search()
# print("Total cost after: " + str(obj.total_cost))
# # obj.show_graph()

import os
import json


directory = 'data/Non_Euclid/Type_1_Small'
files = os.listdir(directory)

results = []
for file in files:
    if file[-3:] != "clt":
        continue
    print("Solving", file)
    result = {}
    result["filename"] = file
    path = os.path.join(directory, file)
    obj = CluSTP(filename=path, graph_type="Non_Euclid")
    obj.init_solution()
    result["init_cost"] = str(obj.total_cost)
    obj.search()
    result["final_cost"] = str(obj.total_cost)

    results.append(result)

with open("result.json", 'w') as f:
    json.dump(results, f)