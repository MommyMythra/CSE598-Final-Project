import numpy as np
from moga import runMoga
from mopso import runMopso
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.indicators.igd_plus import IGDPlus
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


def standardizeData(data, func, algo, mopsoOut=None, mogaOut=None):
    if func not in data:
        data[func] = {}
        if algo not in data[func]:
            data[func][algo] = {
                    "finalSols": [],
                    "finalFronts": [],
                    "runtimes": [],
                    "histories": []
                    }
    if mopsoOut is not None:
        result, run, arch = mopsoOut
        if isinstance(result,tuple):
            sols,fits = result
            finalSols = np.array(sols)
            finalFits = np.array(fits)
        else:
            print("This ain't right")
            return
        data[func][algo]["finalSols"].append(finalSols)
        data[func][algo]["finalFronts"].append(finalFits)
        data[func][algo]["runtimes"].append(run)
        data[func][algo]["histories"].append(arch)
    if mogaOut is not None:
        finalSol = mogaOut.opt.get("X")
        finalFront = mogaOut.opt.get("F")
        run = mogaOut.exec_time

        history = []
        if hasattr(mogaOut, 'history') and mogaOut.history is not None:
            for gen in mogaOut.history:
                F = gen.opt.get("F")
                history.append((gen.n_gen, F))
        data[func][algo]["finalSols"].append(finalSol)
        data[func][algo]["finalFronts"].append(finalFront)
        data[func][algo]["runtimes"].append(run)
        data[func][algo]["histories"].append(history)


def processData(data, refFront):
    process = {}
    for funcName, algo in data.items():
        process[funcName] = {}

        ref = refFront[funcName]
        for algoName, runs, in algo.items():
            runtimes = runs["runtimes"]
            finalFront = runs["finalFronts"]
            finalSol = runs["finalSols"]
            hist = runs["histories"]
            
            avgRun = sum(runtimes) / len(runtimes)

            igdCalc = IGDPlus(ref)
            igdScores = [igdCalc(front) for front in finalFront]

            bestID = int(np.argmin(igdScores))
            bestRunFit = finalFront[bestID]
            bestRunSol = finalSol[bestID]
            bestRunIGD = igdScores[bestID]

            bestHist = hist[bestID]

            igdHist = []
            for (gen,front) in bestHist:
               igdVal = igdCalc(front)
               igdHist.append((gen,igdVal))
            
            process[funcName][algoName] = {
                    "runtimes": runtimes,
                    "avgRun": avgRun,
                    "finalFront": finalFront,
                    "avgFront": np.vstack(finalFront),
                    "bestRunSol": bestRunSol,
                    "bestRunFit": bestRunFit,
                    "bestHist": bestHist,
                    "bestRunIGD": bestRunIGD,
                    "igdHist": igdHist
                    }
    return process

# Define function with bounds, decision variable count, objective count, constraint count, and constraint function evaluation. 
functionSetup = {
        "zdt2": (zdt2, [(0,1)]*30, 30, 2, 0, None),
        "kursawe": (kursawe, [(-5,5)]*3, 3, 2, 0, None),
        "tnk": (tnk, [(0,np.pi)]*2, 2, 2, 2, tnkCon)
        }

# Set up the MULTIRUN framing
runCount = int(input("Input Number of Runs per Test Function: "))
genCount = int(input("Input number of Generations per Run: "))

# Set up Data dictionaries
dataCol = {}
for func in ["zdt2", "kursawe", "tnk"]:
    dataCol[func] = {}
    for algo in ["moga", "mopso"]:
        dataCol[func][algo] = {
                "finalSols" : [],
                "finalFronts": [],
                "runtimes": [],
                "histories": []
                }

for funcName in ["zdt2", "kursawe", "tnk"]:
    testfunc, bounds, decision, objective, conNum, confunc = functionSetup[funcName]
    for i in range(runCount):
        resultGA = runMoga(testfunc, bounds, decision, confunc, nGen = genCount)
        resultPSO = runMopso(testfunc, bounds, decision, objective, conNum, confunc, numGen = genCount)
        standardizeData(
                data = dataCol,
                func = funcName,
                algo = "moga",
                mopsoOut = None,
                mogaOut = resultGA
                )
        standardizeData(
                data = dataCol,
                func = funcName,
                algo = "mopso",
                mopsoOut = resultPSO,
                mogaOut = None
                )


refFront = {
        "zdt2": get_problem("zdt2").pareto_front(),
        "kursawe": get_problem("kursawe").pareto_front(),
        "tnk": get_problem("tnk").pareto_front()
        }

Processed = processData(dataCol, refFront)
print(Processed)
#problem = get_problem("tnk")
#plot(problem.pareto_front(), no_fill=True)

# Access final results
#print("Best decision variables (sample):")
#print(result.X[:5])
#print("Corresponding objectives:")
#print(result.F[:5])
