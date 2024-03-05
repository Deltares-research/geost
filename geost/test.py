import numpy as np
from numpy.polynomial.polynomial import polyfit

x = np.arange(0, 10)
y = np.random.random(10)
deg = 2

data = polyfit(x, y, deg)
