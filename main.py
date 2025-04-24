import numpy as np
from moga import runMoga
from mopso import runMopso

# Schaffer Function at N=2. It will take in a single decision variable
def schaffer(x):
    x = x[0] if isinstance(x, (list, np.ndarray)) else x  # support vector input

    if x <= 1:
        f1 = -x
    elif x <= 3:
        f1 = x - 2
    elif x <= 4:
        f1 = 4 - x
    else:
        f1 = x - 4

    f2 = (x - 5)**2
    return (f1, f2)

# Zitzler-Deb-Thiele Function at N=2
# It will take in a vector of 30 decision variables
def zdt2(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / 29
    f2 = g * (1 - (f1/g)**2)
    return (f1, f2)

# Kursawe Function
# It will take a vector of decision variables
def kursawe(x):
    f1 = 0
    for i in range(0,2):
        temp = -10 * np.exp(-0.2 * np.sqrt(x[i]**2 + x[i+1]**2))
        f1 += temp
    f2 = 0
    for j in range(0,3):
        temp = np.abs(x[i])**0.8 + 5 * np.sin(x[i]**3)
        f2 += temp
    return (f1, f2)

# Viennet Function is diabolical with three objectives and two decision variables.
def viennet(x):
    f1 = 0.5 * (x[0]**2 + x[1]**2) + np.sin(x[0]**2 + x[1]**2)
    f2 = ((3 * x[0] - 2 * x[1] + 4)**2 / 8) + ((x[0] - x[1] + 1)**2 / 27) + 15
    f3 = (1 / (x[0]**2 + x[1]**2 + 1)) - 1.1 * np.exp(-(x[0]**2 + x[1]**2))
    return (f1, f2, f3)

# Poloni Function has two objectives and two decision variables as a vector
def poloni(x):
    A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
    B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
    B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
    f1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    f2 = (x[0] + 3)**2 + (x[1] + 1)**2
    return (f1, f2)

# Define function with bounds, decision variable count, objective count

functionSetup = {
        "Schaffer": (schaffer, [(-5,10)], 1, 2),
        "Zitzler-Deb-Thiele": (zdt2, [(0,1)]*30, 30, 2),
        "Kursawe": (kursawe, [(-5,5)]*3, 3, 2),
        "Viennet": (viennet,[(-3,3)]*2, 2, 3),
        "Poloni": (poloni, [(-np.pi,np.pi)]*2, 2, 2)
        }

testfunc, bounds, decision, objective = functionSetup["Kursawe"]
# Run pymoo
#result = runMoga(testfunc, bounds, decision)
runMopso(testfunc, bounds, decision, objective)

# Access final results
#print("Best decision variables (sample):")
#print(result.X[:5])
#print("Corresponding objectives:")
#print(result.F[:5])
