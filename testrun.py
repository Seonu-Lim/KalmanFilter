import numpy as np
from Filter.filters import LowPassFilter

class TestDataGenerator(object) :
    def __init__(self,shape=(2,2)) :
        self.shape = shape
    def __iter__(self) :
        return self
    def __next__(self) :
        return self.next()
    def next(self) :
        return np.random.rand(*self.shape)



datagen = TestDataGenerator((10,10,3))
LPF = LowPassFilter(datagen.next())
for i in range(10) :
    LPF.update(datagen.next())
    print(LPF.estimate)
    print("=========================")
