"""Base class for Reinforcement Agents
"""

from BiSUNA.parameters import ParameterType

class Reinforcement_Agent:
    def __init__(self):
        self.action = []
        self.number_of_observation_vars = self.number_of_action_vars = 0

    def init(self, number_of_observation_vars, number_of_action_vars):
        pass

    def step(self, observation, reward):
        pass

    def stepBestAction(self, observation):
        pass

    def saveAgent(self, filename):
        pass

    def loadAgent(self, filename):
        pass

    def endEpisode(self, reward):
        pass
    