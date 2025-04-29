import numpy as np
from moga import runMoga
from mopso import runMopso
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.indicators.igd_plus import IGDPlus
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        #print(result)
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
            #print(len(finalFront))
            igdScores = [igdCalc(front) for front in finalFront]
            #print(finalFront)
            #print(igdScores)

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
                    "igdHist": igdHist,
                    "igdScores":igdScores
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
        print(f'Current Run for function {funcName}: {i}')
        resultGA = runMoga(testfunc, bounds, decision, confunc, nGen = genCount)
        resultPSO = runMopso(testfunc, bounds, decision, objective, conNum, confunc, numGen = genCount)
        #print(resultPSO)
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


####################################################### THE GRAPHING AND OUTPUT FOR THIS BAD BOY ####################################################################
def plot_runtime(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        y = data['runtimes']
        ax.scatter(range(len(y)), y, label=f'{algo} runtimes')
        ax.plot([-0.5, len(y)-0.5], [data['avgRun'], data['avgRun']], linestyle='--', label=f'{algo} avg')
    ax.set_title(f'Runtime Comparison - {func}')
    ax.set_xlabel('Run Index')
    ax.set_ylabel('Time (s)')
    ax.legend()
    return fig

def plot_best_fronts(processed, ref, func):
    fig, ax = plt.subplots()
    ax.scatter(ref[func][:,0], ref[func][:,1], c='k', marker='x', label='Reference PF')
    for algo, data in processed[func].items():
        ax.scatter(data['bestRunFit'][:,0], data['bestRunFit'][:,1], label=algo)
    ax.set_title(f'Best Frontiers - {func}')
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    return fig

def plot_igd_convergence(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        gens, vals = zip(*data['igdHist'])
        ax.scatter(gens, vals, label=algo)
    ax.set_title(f'IGD+ Convergence - {func}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('IGD+')
    ax.legend()
    return fig

def plot_all_igd(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        y = data['runtimes']
        print(data['igdScores'])
        ax.scatter(range(len(y)), data['igdScores'], label=f'{algo}' if i==0 else None, alpha=0.5)
    ax.set_title(f'Final IGD+ Scores - {func}')
    ax.set_xlabel('Run Index')
    ax.set_ylabel('IGD+')
    ax.legend()
    return fig

# ========== GUI SETUP ==========

def embed_plot(tab, fig):
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def build_gui(processed, reference_fronts):
    root = tk.Tk()
    root.title("Multi-Objective Optimization Dashboard")
    notebook = ttk.Notebook(root)

    categories = ["Runtimes", "Pareto Frontiers", "IGD+ Graphs"]
    plot_funcs = {
        "Runtimes": [plot_runtime],
        "Pareto Frontiers": [plot_best_fronts],
        "IGD+ Graphs": [plot_igd_convergence, plot_all_igd],
    }

    for category in categories:
        cat_tab = ttk.Notebook(notebook)
        notebook.add(cat_tab, text=category)

        for func in processed.keys():
            func_tab = ttk.Notebook(cat_tab)
            cat_tab.add(func_tab, text=func)

            for plot_func in plot_funcs[category]:
                plot_tab = ttk.Frame(func_tab)
                func_tab.add(plot_tab, text=plot_func.__name__.replace("plot_", "").replace("_", " ").title())

                if 'ref' in plot_func.__code__.co_varnames:
                    fig = plot_func(processed, reference_fronts, func)
                else:
                    fig = plot_func(processed, func)
                embed_plot(plot_tab, fig)

    notebook.pack(expand=1, fill="both")
    root.mainloop()
build_gui(Processed, refFront)

