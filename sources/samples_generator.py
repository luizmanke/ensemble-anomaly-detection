# reference: https://github.com/bzantium/OCSVM-hyperparameter-selection

# System modules
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SelfAdaptiveShifting:

    def __init__(self, data):
        self.data = data
        self.k = math.ceil(5 * math.log10(len(self.data)))
        self.distances = None
        self.all_neighbor_indices = None
        self.edge_indice = None
        self.normal_vectors = None
        self.pseudo_outliers = None
        self.pseudo_targets = None

    def edge_pattern_detection(self, threshold=0.01):
        edge_indice, normal_vectors = [], []
        model = NearestNeighbors(n_neighbors=self.k + 1).fit(self.data)
        self.distances, self.all_neighbor_indices = model.kneighbors(self.data)
        for i in range(len(self.data)):
            neighbors = self.data[self.all_neighbor_indices[i][1:]]
            x_i = np.tile(self.data[i], (self.k, 1))
            vectors = \
                (x_i - neighbors) / (np.expand_dims(self.distances[i][1:], axis=1) + 1e-8)
            vectors_norm = np.sum(vectors, axis=0, keepdims=True)
            normal_vectors.append(vectors_norm.squeeze())
            theta = np.dot(vectors, vectors_norm.transpose())
            if np.mean(theta >= 0) > 1 - threshold:
                edge_indice.append(i)
        self.normal_vectors = np.array(normal_vectors)
        self.edge_indice = edge_indice

    def generate_pseudo_outliers(self):
        edge_data = self.data[self.edge_indice]
        distances_mean = np.mean(self.distances[self.edge_indice][1:])
        edge_normal_vectors = self.normal_vectors[self.edge_indice]
        self.pseudo_outliers = (
            edge_data + edge_normal_vectors /
            (np.linalg.norm(edge_normal_vectors, axis=1, keepdims=True) + 1e-8) *
            distances_mean
        )
        return self.pseudo_outliers

    def generate_pseudo_targets(self):
        shift_directions = -1 * self.normal_vectors
        unit_shift_directions = (
            shift_directions /
            (np.linalg.norm(shift_directions, axis=1, keepdims=True) + 1e-8)
        )
        pseudo_targets = []
        for i in range(len(self.data)):
            min_product = self._get_product_minimum(
                self.data[i], unit_shift_directions[i], self.all_neighbor_indices[i])
            pseudo_targets.append(self.data[i] + min_product * unit_shift_directions[i])
            pseudo_targets.append(self.data[i] - min_product * unit_shift_directions[i])
        self.pseudo_targets = np.array(pseudo_targets)
        return self.pseudo_targets

    def _get_product_minimum(self, x_i, unit_shift_direction, neighbor_indices):
        minimum_x_ij = x_i
        minimum_product = float("inf")
        for neighbor_index in neighbor_indices[1:]:
            x_ij = self.data[neighbor_index]
            current_product = np.dot(unit_shift_direction, x_ij - x_i)
            if 0 < current_product < minimum_product:
                minimum_product = current_product
                minimum_x_ij = x_ij
        if minimum_x_ij is x_i:
            minimum_product = 0
        return minimum_product
