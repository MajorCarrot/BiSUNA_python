"""Agent controlled by humans (Dummy Agent)
"""

from BiSUNA.parameters import *
from BiSUNA.agents.Reinforcement_Agent import *
from BiSUNA.environments.Reinforcement_Environment import *

class Dummy(Reinforcement_Agent):
    def __init__(self, env):
        assert isinstance(env, Reinforcement_Environment)
        self.env = env

    def init(self, number_of_observation_vars, number_of_action_vars):
        self.number_of_observation_vars = number_of_action_vars
        self.number_of_action_vars = number_of_action_vars

        self.action = [None] * self.number_of_action_vars

    def step(self, observation, reward):
        print(self.env)

        print(observation, self.number_of_observation_vars)

        print(f"Reward {reward}")

        for i in range(self.number_of_action_vars):
            print(f"Action {i}")
            if CONTINUOUS_PARAM:
                self.action[i] = float(input())
            else:
                self.action[i] = int(input())

        print(self.action, self.number_of_action_vars)

    def stepBestAction(self, observation):
        return 0

    def endEpisode(self, reward):
        print("Episode finished!! Time for a beer!!")

    def saveAgent(self, filename):
        pass

    def loadAgent(self, filename):
        pass
