import numpy as np


def get_data_non_euclid(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n = int(lines[2][:-1].split(":")[1])  # number of vertices
    n_clusters = int(lines[3][:-1].split(":")[1])

    # coordinates of vertices
    # coordinates = []
    # for i in range(6, len(lines) - n_clusters - 3):
    #     numbers = lines[i][:-1].replace("  ", " ").split(' ')
    #     coordinates.append((int(numbers[1]), int(numbers[2])))
    distances = np.zeros([n, n])
    for i in range(6, len(lines) - n_clusters - 3):
        line = " ".join(lines[i][:-1].split())
        numbers = line.split()
        for j, number in enumerate(numbers):
            distances[i-6, j] = number

    # R[i]: set of vertices in cluster i
    R = []
    for i in range(len(lines) - n_clusters - 1, len(lines) - 1):
        Ri = set([int(vertex) for vertex in lines[i][:-1].split()[1:-1]])
        R.append(Ri)

    # source vertice
    source_vertex = int(lines[len(lines) - n_clusters - 2][:-1].split(":")[1])

    return n, n_clusters, distances, R, source_vertex

def get_data_euclid(filename):
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


def get_result(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    size = len(lines) - 6
    X = np.zeros([size, size])
    for i in range(6, len(lines)):
        line = lines[i].split(" ")[:-1]
        for j in range(size):
            if line[j] == '1':
                X[i-6, j] = 1
    return X

if __name__ == '__main__':
    res = get_data_non_euclid('data/Non_Euclid/Type_1_Small/5berlin52.clt')
    print(res[2])
