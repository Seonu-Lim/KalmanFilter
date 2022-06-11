import numpy as np
from numpy.linalg import multi_dot,pinv


class LowPassFilter():
    def __init__(self,initial_point,alpha=0.1) :
        self.observation = initial_point
        self.estimate = initial_point
        self.alpha = alpha

    def update(self,new_observation) :
        self.estimate = self.estimate * self.alpha + new_observation * (1-self.alpha)

class KalmanFilter():
    def __init__(self,initial_point,P,A,Q,H) :
        self.observation = initial_point
        self.pred = initial_point
        self.estimate = initial_point

        self.P_pred = P
        self.P_est = P
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.K  = 0


    def prediction(self) :
        self.pred = np.dot(self.A,self.estimate)
        self.P_pred = multi_dot([self.A,self.P_est,self.A.T]) + self.Q

    def calculate_kalman_gain(self) :
        tm = multi_dot([self.H,self.P_pred,self.H.T]) + self.R
        self.K = multi_dot([self.P_pred,self.H.T,pinv(tm)])

    def estimation(self,observation) :
        self.estimate = self.prediction + np.dot(self.K,observation-np.dot(self.H,self.prediction))
        self.P_est = self.P_pred - multi_dot([self.K,self.H,self.P_pred])

    def update(self,observation) :

        self.prediction()
        self.calculate_kalman_gain()
        self.estimation(observation)

        return self.estimate

