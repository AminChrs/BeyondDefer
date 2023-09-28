# define binary entropy function

import numpy as np

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

# find 2x(1-x)+(x^2+(1-x)^2)*H(x^2/(2x^2-2x+1))-H(x)

def f(x):
    return 2*x*(1-x) + (x**2+(1-x)**2)*entropy(x**2/(2*x**2-2*x+1)) - entropy(x)
    

# plot f(x) for x in [0,1/2]

import matplotlib.pyplot as plt

x = np.linspace(0, 1/2, 100)
plt.plot(x, f(x))
plt.show()

# find -2x(1-x)+(x^2+(1-x)^2)*(log2(x^2+(1-x)^2))

def g(x):
    return 2*x*(1-x) + (x**2+(1-x)**2)*(np.log2(x**2+(1-x)**2))

# plot g(x) for x in [0,1/2]

plt.plot(x, g(x))
plt.show()

# find -2x(1-x)+(x^2+(1-x)^2)*H(x^2/(2x^2-2x+1))

def h(x):
    return -2*x*(1-x) + (x**2+(1-x)**2)*entropy(x**2/(2*x**2-2*x+1))

# plot h(x) for x in [0,1/2]

plt.plot(x, h(x))
plt.show()

# find -2x(1-x)+(x^2+(1-x)^2)*log(x^2/(2x^2-2x+1))*log(1-(x^2/(2x^2-2x+1)))

def i(x):
    return -2*x*(1-x) + (x**2+(1-x)**2)*np.log2(x**2/(2*x**2-2*x+1))*np.log2(1-(x**2/(2*x**2-2*x+1)))

# plot i(x) for x in [0,1/2]

plt.plot(x, i(x))
plt.show()

# find -2x(1-x)-(x^2+(1-x)^2)*log(x^2/(2x^2-2x+1))

def j(x):
    return -2*x*(1-x) - (x**2+(1-x)**2)*np.log2(x**2/(2*x**2-2*x+1))

# plot j(x) for x in [0,1/2]

plt.plot(x, j(x))
plt.show()

# find -2x(1-x)+log(x^2/(2x^2-2x+1))*log(1-(x^2/(2x^2-2x+1)))

def k(x):
    return -2*x*(1-x) + np.log2(x**2/(2*x**2-2*x+1))*np.log2(1-(x**2/(2*x**2-2*x+1)))

# plot k(x) for x in [0,1/2]

plt.plot(x, k(x))
plt.show()
