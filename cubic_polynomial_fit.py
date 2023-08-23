import math
import numpy as np

from scipy.optimize import curve_fit

def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

class CubicPolynomial1D:
    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficients
        popt, pcov = curve_fit(cubic_func, x, y, check_finite=True)
        self.a, self.b, self.c, self.d = popt
        self.score = np.linalg.cond(pcov)

    def calc_position(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        position = cubic_func(x, self.a, self.b, self.c, self.d)

        return position

    def calc_first_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        dy = 3 * self.a * x**2 + 2 * self.b * x + self.c
        return dy

    def calc_second_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        ddy = 6 * self.a * x + 2 * self.b
        return ddy


class CubicPolynomial2D:

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicPolynomial1D(self.s, x)
        self.sy = CubicPolynomial1D(self.s, y)
        self.score_x = self.sx.score
        self.score_y = self.sy.score
        self.score = self.score_x * self.score_y

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw
