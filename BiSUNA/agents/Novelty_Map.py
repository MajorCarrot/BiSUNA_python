import math
from dataclasses import dataclass

from BiSUNA.parameters import CONTINUOUS_PARAM, ParameterType


@dataclass
class cell:
    weight: list = []
    frequency: int = None
    pointer: list = []


class Novelty_Map:
    def __init__(self, population_size, number_of_inputs):

        self.population_size = population_size
        self.map = [cell()]
        self.number_of_inputs = number_of_inputs

        for i in range(population_size):
            self.map[i].weight = [0] * self.number_of_inputs

        self.current_population_size = 0
        self.min_disatance = ParameterType(None)
        self.worst_individual = None

    def insertIndividual(self, individual_index, weight):
        for i in range(self.number_of_inputs):
            self.map[individual_index].weight[i] = weight[i]

        self.map[individual_index].frequency = 0

    def updateMinDistance(self):
        self.worst_individual = 0
        self.min_disatance = self.minDistance(self.worst_individual)

        for i in range(1, self.current_population_size):
            min_distance_i = self.minDistance(i)
            if self.min_disatance > min_distance_i:
                self.min_disatance = min_distance_i
                self.worst_individual = i

    def minDistance(self, individual=None, input_array=[]) -> ParameterType:
        if individual is not None:
            if individual != 0:
                dist = self.quadraticDist(individual, individual_b=0)
            else:
                dist = self.quadraticDist(individual, individual_b=1)

            for i in range(1, self.current_population_size):
                if i != individual and dist > self.quadraticDist(
                    individual, individual_b=i
                ):
                    dist = self.quadraticDist(individual, individual_b=i)

            return dist

        if input_array == []:
            raise ValueError(
                "Either individual or input_array must be specified"
            )

        dist = self.quadraticDist(0, input_array=input_array)
        closest_individual = 0
        for i in range(1, self.current_population_size):
            qd = self.quadraticDist(i, input_array=input_array)
            if dist > qd:
                dist = qd
                closest_individual = i

        return closest_individual, dist

    def quadraticDist(self, individual_a, individual_b=None, input_array=[]):
        if input_array != []:
            dist = 0
            if CONTINUOUS_PARAM:
                for i in range(self.number_of_inputs):
                    dist += (
                        self.map[individual_a].weight[i] - input_array[i]
                    ) ** 2
            else:
                for i in range(self.number_of_inputs):
                    w = self.map[individual_a].weight[i]
                    ia = input_array[i]
                    dist += self.hamming_distance(w, ia)
            return dist

        if individual_b is None:
            raise ValueError(
                "Either individual_b or input_array must be specified"
            )

        dist = 0
        if CONTINUOUS_PARAM:
            for i in range(self.number_of_inputs):
                dist += (
                    self.map[individual_a].weight[i]
                    - self.map[individual_b.weight[i]]
                ) ** 2

        else:
            for i in range(self.number_of_inputs):
                ia = self.map[individual_a].weight[i]
                ib = self.map[individual_b.weight[i]]
                dist += self.hamming_distance(ia, ib)

        return dist

    def hamming_distance(self, x, y):
        if CONTINUOUS_PARAM:
            return 0
        else:
            return bin(x ^ y).count("1")

    def __repr__(self):
        s = ""
        for i in range(self.current_population_size):
            for j in range(self.number_of_inputs):
                s += f"{self.map[i].weight[j]} "

        return s.strip()

    def inp(self, input_array):
        if self.current_population_size < self.population_size:
            inserted_index = self.current_population_size
            self.insertIndividual(inserted_index, input_array)
            self.map[inserted_index].pointer = None
            self.current_population_size += 1

            if self.current_population_size == self.population_size:
                self.updateMinDistance()

            return inserted_index

        closest_individual, current_input_min_distance = ParameterType(
            self.min_disatance(input_array=input_array)
        )

        if self.min_disatance < current_input_min_distance:
            self.insertIndividual(self.worst_individual, input_array)
            return self.worst_individual

        return closest_individual

    def inputNeutral(self, input_array):
        closest_individual, _ = self.minDistance(input_array=input_array)
        return closest_individual
