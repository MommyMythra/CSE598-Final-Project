import numpy as np
from moga import runMoga
from mopso import runMopso
from pymoo.problems import get_problem
from pymoo.util.plotting import plot

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

# Tanaka, or TNK, is a twin minimization function with complex constraint space
def tnk(x):
    f1 = x[0]
    f2 = x[1]
    return (f1, f2)

# Tanaka Constraint funciton
def tnkCon(x):
    arctan = np.arctan2(x[0],x[1])
    c1 = -(x[0]**2 + x[1]**2 - 1 - (0.1 * np.cos(16 * arctan)))
    c2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.5
    return (c1, c2)


# Define function with bounds, decision variable count, objective count, constraint count, and constraint function evaluation. 

functionSetup = {
        "Zitzler-Deb-Thiele": (zdt2, [(0,1)]*30, 30, 2, 0, None),
        "Kursawe": (kursawe, [(-5,5)]*3, 3, 2, 0, None),
        "Tanaka": (tnk, [(0,np.pi)]*2, 2, 2, 2, tnkCon)
        }

testfunc, bounds, decision, objective, conNum, confunc = functionSetup["Tanaka"]
# Set up the MULTIRUN framing
runCount = int(input("Input Number of Runs per Test Function: "))
genCount = int(input("Input number of Generations per Run: "))
for j in range(3):
    if j == 0:
        testfunc, bounds, decision, objective, conNum, confunc = functionSetup["Zitzler-Deb-Thiele"]
    elif j == 1:
        testfunc, bounds, decision, objective, conNum, confunc = functionSetup["Kursawe"]
    elif j == 2:
        testfunc, bounds, decision, objective, conNum, confunc = functionSetup["Tanaka"]
    for i in range(runCount):
        result = runMoga(testfunc, bounds, decision, confunc, nGen = genCount)
        runMopso(testfunc, bounds, decision, objective, conNum, confunc, numGen = genCount)


#problem = get_problem("tnk")
#plot(problem.pareto_front(), no_fill=True)

# Access final results
#print("Best decision variables (sample):")
#print(result.X[:5])
#print("Corresponding objectives:")
#print(result.F[:5])
