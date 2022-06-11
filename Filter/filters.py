import numpy as np


class LowPassFilter():
    def __init__(self,initial_point,alpha=0.1) :
        self.observation = initial_point
        self.estimate = initial_point
        self.alpha = alpha

    def update(self,new_observation) :
        self.estimate = self.estimate * self.alpha + new_observation * (1-self.alpha)



