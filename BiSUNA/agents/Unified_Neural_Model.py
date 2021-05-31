from dataclasses import dataclass

from BiSUNA.agents.modules.ModuleDebug import *
from BiSUNA.agents.modules.Module import *
from BiSUNA.agents.Novelty_Map import *
from BiSUNA.agents.Reinforcement_Agent import *
from BiSUNA.parameters import *


@dataclass
class nmap_cell(Reinforcement_Agent):
    module: Module = None
    fitness: float = None


class UnifiedNeuralModel:
    def __init__(self, random):
        self.random = random
        self.testing_subpop = 0
        self.testing_individual = 0
        self.generation = 1
        self.best_index = 0

        self.nmap = Novelty_Map(NOVELTY_MAP_SIZE, SPECTRUM_SIZE)

        if NOVELTY_MAP_SIZE >= SUBPOPULATION_SIZE:
            raise ValueError(
                "Novelty map's size is bigger than or equal to the subpopulation size"
            )

        if NUMBER_OF_SUBPOPULATIONS != 1:
            raise ValueError("Number of subpopulations is different than 1")

        self.step_counter = 0

    def printBest(self):
        pass

    def spectrumDiversityEvolve(self):
        pass

    def subpopulationObjective(self, module, fitness, subpopulation_index):
        pass

    def __repr__(self):
        pass

    def init(self, number_of_observation_vars, number_of_action_vars):
        self.number_of_observation_vars = number_of_observation_vars
        self.number_of_action_vars = number_of_action_vars

        self.action = [ParameterType(0)] * self.number_of_action_vars

        self.subpopulation = [
            [None] * SUBPOPULATION_SIZE
        ] * NUMBER_OF_SUBPOPULATIONS
        self.tmp_subpopulation = [
            [None] * SUBPOPULATION_SIZE
        ] * NUMBER_OF_SUBPOPULATIONS

        self.fitness = [[None] * SUBPOPULATION_SIZE] * NUMBER_OF_SUBPOPULATIONS
        self.tmp_fitness = [
            [None] * SUBPOPULATION_SIZE
        ] * NUMBER_OF_SUBPOPULATIONS

        for i in range(NUMBER_OF_SUBPOPULATIONS):
            for j in range(SUBPOPULATION_SIZE):
                self.subpopulation[i][j] = Module(
                    self.number_of_observation_vars,
                    self.number_of_action_vars,
                    INITIAL_ALLOCATION_LENGTH,
                )
                self.tmp_subpopulation[i][j] = Module(
                    self.number_of_observation_vars,
                    self.number_of_action_vars,
                    INITIAL_ALLOCATION_LENGTH,
                )
                self.fitness[i][j] = EXTREME_NEGATIVE_REWARD

                for _ in range(NUMBER_OF_INITIAL_MUTATIONS):
                    self.subpopulation[i][j].structuralMutation()
                
                self.subpopulation[i][j].updatePrimerList()

    def step(self, observation, reward):
        pass

    def stepBestAction(self, observation):
        pass

    def calculateSpectrum(
        self, spectrum, subpopulation_index, individual_index
    ):
        pass

    def endBestEpisode(self):
        pass

    def endEpisode(self, reward):
        pass

    def saveAgent(self, filename):
        pass

    def loadAgent(self, filename):
        pass

    def printSubpop(self):
        pass
