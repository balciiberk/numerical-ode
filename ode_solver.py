import numpy as np
import matplotlib.pyplot as plt

class Solver:
    '''A class that solves first order ordinary differential equations (ODE) by using Euler or Runge-Kutta methods.

    Parameters
    ----------
    function : function
        The function f(x,y) where the ODE is written in the form of dy/dx = f(x,y) 
    boundary : list of float or int
        The boundary conditions of x and y in the same order with the function.
    final_x : float or int
        The last x value of iteration.
    step_size : float or int
        Step size of iteration.
    
    Methods
    -------
    euler():
        Solves ODE with Euler method and returns all the x and y values of iterations.
    heun():
        Solves ODE with Heun's method and returns all the x and y values of iterations.
    ralston():
        Solves ODE with Ralston's method and returns all the x and y values of iterations.
    midpoint():
        Solves ODE with midpoint method and returns all the x and y values of iterations.
    runge2nd():
        Solves ODE with second Runge-Kutta with a custom a2 value. heun(), ralston(), midpoint() methods
        use this method with a2 values 1/2, 2/3 and 1 respectively. It returns all the x and y values of iterations.
        classic4th():
        Solves ODE with fourth order Runge-Kutta methdod and and returns all the x and y values of iterations.
    compare():
        Takes as many methods as wanted in string format(i.e. .compare_method('euler()', 'runge2nd(3/4)'),
        plots the comparison graph and returns the values in dictionary format.
    
    TODO
    ----
    - Add higher order differential equations
    
    References
    ----------
    [1] https://nm.mathforcollege.com/
    '''
    def __init__(self, function, boundary, final_x, step_size):
        self.f = function
        self.boundary = boundary
        self.h = step_size
        self.final_x = final_x
    
    def euler(self):
        x = np.linspace(self.boundary[0], self.final_x, int(np.abs((self.final_x - self.boundary[0])/self.h) + 1))
        y = np.empty(len(x))
        y[0] = self.boundary[1]
        for i in range(1, len(x)):
            y[i] = y[i-1] + self.f(x[i-1], y[i-1]) * self.h
        return x,y
            
    def runge2nd(self, a2):
        a1 = 1 - a2
        p = 1/(2*a2)
        q = p
        
        x = np.linspace(self.boundary[0], self.final_x, int(np.abs((self.final_x - self.boundary[0])/self.h) + 1))
        y = np.empty(len(x))
        y[0] = self.boundary[1]
        
        for i in range(1, len(x)):
            y[i] = y[i-1] + (a1 * self.f(x[i-1], y[i-1]) + \
                             a2 * self.f(x[i-1] + p * self.h, y[i-1] + q * self.f(x[i-1], y[i-1]) * self.h) ) * self.h
        return x,y
        
    def heun(self):
        return self.runge2nd(1/2)
    
    def ralston(self):
        return self.runge2nd(2/3)
    
    def midpoint(self):
        return self.runge2nd(1)
    
    def classic4th(self):
        x = np.linspace(self.boundary[0], self.final_x, int(np.abs((self.final_x - self.boundary[0])/self.h) + 1))
        y = np.empty(len(x))
        y[0] = self.boundary[1]
        
        for i in range(1, len(x)):
            k1 = self.f(x[i-1], y[i-1])
            k2 = self.f(x[i-1] + self.h/2, y[i-1] + k1*self.h/2)
            k3 = self.f(x[i-1] + self.h/2, y[i-1] + k2*self.h/2)
            k4 = self.f(x[i-1] + self.h, y[i-1] + k3*self.h)
            y[i] = y[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)*self.h
            
        return x,y
    
    def compare(self, *args):
        for i in args:
            if not i.split('(')[0] in dir(self):
                raise Exception(f'There is no method called \'{i}\'')
        results = {} 
        for i in args:
            result = eval('self.' + i)
            plt.plot(*result,label = i)
            results[i] = result
        plt.legend()
        plt.show()
        return results
    

