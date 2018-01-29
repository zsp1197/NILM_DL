import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Parameters(object):
    def __init__(self):
        self.penalty = 999999
        self.considered_appliances = []
        # self.considered_appliances=['dish washer','fridge']
        # self.considered_appliances=['dish wahser','fridge','microwave']
        self.max_T = pd.Timedelta('30s')

        Y = np.array([150, 100, 80, 50, 40, 30, 20, 10, 6, 5, 2])
        X = np.array([1500, 1000, 700, 500, 400, 200, 100, 50, 10, 4, 0])
        self.p = np.poly1d(np.polyfit(X, Y, 3))
        self.sax_steps = self.get_sax_step()
        self.labeling_window_size = pd.Timedelta('2S')
        self.delta_detection=10
        self.sample_T=pd.Timedelta('1S')

    def get_sax_step(self):
        step = 0
        result = []
        while (step < 7000):
            result.append(step)
            step = step + self.p(step)
        result = result[0:-1:2]
        return result
