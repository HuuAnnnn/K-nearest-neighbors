import math


class KNN():
    def __init__(self, k_neighbors: int, p: int) -> None:
        self.k = k_neighbors
        self.p = p

    def load_data(self, dataset: list, label: list):
        self.labels = label
        self.dataset = dataset

    def predict(self, vectors):
        result = []
        for vector in vectors:
            result.append(self.__vec_label(vector))

        return result

    def __vec_label(self, vec):
        neighbors = []
        for i in range(len(self.dataset)):
            neighbors.append((self.__distance(
                vec, self.dataset[i]), self.labels[i]))

        neighbors_asc = sorted(neighbors, key=lambda x: x[0])

        return self.__most_label(neighbors=neighbors_asc[:self.k])

    def __most_label(self, neighbors):
        labels = {}
        for label in self.labels:
            labels[label] = 0

        for neighbor in neighbors:
            labels[neighbor[1]] += 1

        sorted_labels = sorted(
            labels.items(), key=lambda x: x[1], reverse=True)

        return sorted_labels[0][0]

    def __distance(self, vector1, vector2):
        return self.__norm(vector1, vector2, self.p)

    def __norm(self, vec1, vec2, p):
        sum = 0
        for i in range(len(vec2)):
            sum += (math.fabs(vec2[i] - vec1[i]))**p

        return sum**(1/p)
