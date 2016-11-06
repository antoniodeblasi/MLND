'''
Created on Oct 23, 2016

@author: antoniodeblasi
'''
from collections import defaultdict



class Action:
    def __init__(self):
        self.id = None
        self.value = None

class Teacher:
    def __init__(self, actionIds):
        self.rewards = dict()
    
    def give_reward(self, action):
        self.rewards[action.id].get_reward();
        

class Agent:
    def __init__(self, policy, actions):
        self.policy = policy
        self.actions = actions
    
    def select_action(self):
        self.policy.select_action()
        

class Policy:
    def __init__(self, actions):
        self.actions = actions
    
    def select_action(self):
        return max(self.actions)
        
        

class ActionValue:
    def __init__(self):
        self.value = 0.0
        self.n = 0
    
    def update(self, reward):
        self.value += (reward - self.value)/(self.n + 1) 

class SampleAverage(ActionValue):
    def __init__(self):
        super(SampleAverage, self).__init__()
        self._actionValueMap = defaultdict(lambda: 0.0)
        self._n = 0
    
    def update(self, reward, action):
        ActionValue.update(self, reward, action)
        try:
            current_mean = self.actionValueMap[action]
        except KeyError:
            current_mean = 0.0
        
        self.actionValueMap[action] = current_mean + (reward - current_mean)/(self._n + 1)
        
        
class Simulation:
    def __init__(self, agent, teacher, actions, time_steps):
        self.agent      = agent
        self.actions    = actions
        self.teacher    = teacher
        self.time_steps = time_steps
        self.total_expected_reward = 0.0
    
    def run(self):
        for _ in range(0, self.time_steps):
            action = self.agent.select_action()
            reward = self.teacher.give_reward(action)
            self.total_expected_reward += reward
            self.agent.policy.update(action, reward)
        
        