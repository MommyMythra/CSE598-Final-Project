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
        temp = np.abs(x[j])**0.8 + 5 * np.sin(x[j]**3)
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


# This function will take all the ugly, variable data and convert it
# to a standard form for processing
# Input:
#       data: The culminating structure set up elsewhere
#       func: The name of the test function
#       algo: The name of the algorithm (MOGA/MOPSO)
#       mopsoOut: The output data for the MOPSO. None default
#       mogaOut: The output data for the MOGA. None default
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

# Takes the standardized data and processes it into a secondary form for graphing. Some information may not be used
# Input:
#       data: The standardized data structure
#       refFront: The reference problem to get the reference front from Pymoo
# Return:
#       process: The processed data structure
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

# Run the algorithms
for funcName in ["zdt2", "kursawe", "tnk"]:
    testfunc, bounds, decision, objective, conNum, confunc = functionSetup[funcName]
    for i in range(runCount):
        print(f'Current Run for function {funcName}: {i}')
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


# Set up the reference fronts
refFront = {
        "zdt2": get_problem("zdt2").pareto_front(),
        "kursawe": get_problem("kursawe").pareto_front(),
        "tnk": get_problem("tnk").pareto_front()
        }

# Obtain the processed data
Processed = processData(dataCol, refFront)


####################################################### THE GRAPHING AND OUTPUT FOR THIS BAD BOY ####################################################################
# Plots the runtime plots for comparison
# Input:
#       processed: the processed data structure
#       func: The test function being drawn
# Return:
#       fig: The generated figure
def plotRuntime(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        y = data['runtimes']
        ax.scatter(range(len(y)), y, label=f'{algo} runtimes')
        ax.plot([-0.5, len(y)-0.5], [data['avgRun'], data['avgRun']], linestyle='--', label=f'{algo} avg')
    ax.set_title(f'Runtime Comparison - {func}')
    ax.set_xlabel('Run Index')
    ax.set_ylabel('Time (s)')
    ax.legend()
    fig.savefig(f"output/{func}_runtime.png", bbox_inches="tight")
    return fig

# Plots the best pareto frontiers from each algo and the reference front
# Input:
#       processed: the processed data structure
#       ref: the reference frontier
#       func: The test function being drawn
# Return:
#       fig: The generated figure
def plotBestFronts(processed, ref, func):
    fig, ax = plt.subplots()
    ax.scatter(ref[func][:,0], ref[func][:,1], c='k', marker='x', label='Reference PF')
    for algo, data in processed[func].items():
        ax.scatter(data['bestRunFit'][:,0], data['bestRunFit'][:,1], label=algo)
    ax.set_title(f'Best Frontiers - {func}')
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.legend()
    fig.savefig(f"output/{func}_front.png", bbox_inches="tight")
    return fig

# Plots the IGD+ values calculated earlier for comparison. Specifically the generation by generation
# values for each best run
# Input:
#       processed: the processed data structure
#       func: The test function being drawn
# Return:
#       fig: The generated figure
def plotIGDConvergence(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        gens, vals = zip(*data['igdHist'])
        ax.scatter(gens, vals, label=algo)
    ax.set_title(f'IGD+ Convergence - {func}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('IGD+')
    ax.legend()
    fig.savefig(f"output/{func}_IGDConverge.png", bbox_inches="tight")
    return fig

# Plots every run's IGD+ value for comparison.
# Input:
#       processed: the processed data structure
#       func: The test function being drawn
# Return:
#       fig: The generated figure
def plotAllIgd(processed, func):
    fig, ax = plt.subplots()
    for algo, data in processed[func].items():
        y = data['runtimes']
        print(data['igdScores'])
        ax.scatter(range(len(y)), data['igdScores'], label=f'{algo}' if i==0 else None, alpha=0.5)
    ax.set_title(f'Final IGD+ Scores - {func}')
    ax.set_xlabel('Run Index')
    ax.set_ylabel('IGD+')
    ax.legend()
    fig.savefig(f"output/{func}_allIGD.png", bbox_inches="tight")
    return fig

# The generation of the summary text for reporting later
# Input:
#       processed: The processed data 
#       func: The test function
# output:
#       The summary text
def generateSummaryText(processed, func):
    lines = [f"Summary for {func}:\n"]
    for algo, data in processed[func].items():
        best_runtime = min(data['runtimes'])
        avg_runtime = data['avgRun']
        best_igd = min(data['igdScores'])
        avg_igd = sum(data['igdScores']) / len(data['igdScores'])
        lines.append(
            f"{algo}:\n"
            f"  Best Runtime: {best_runtime:.4f} s\n"
            f"  Average Runtime: {avg_runtime:.4f} s\n"
            f"  Best IGD+: {best_igd:.4f}\n"
            f"  Average IGD+: {avg_igd:.4f}\n"
        )
    return "\n".join(lines)

###################################### GUI SETUP ###############################

# Create a final tab that contains the summary text
# Input:
#       parent: The parent window
#       processed: The processed data
def addSummaryTab(parent, processed):
    summaryTab = ttk.Notebook(parent)
    parent.add(summaryTab, text="Summary Stats")

    for func in processed.keys():
        tab = ttk.Frame(summaryTab)
        summaryTab.add(tab, text=func)

        textBox = tk.Text(tab, wrap="word", font=("Courier", 10))
        textBox.pack(expand=True, fill="both")
        summary = generateSummaryText(processed, func)
        textBox.insert("1.0", summary)
        textBox.config(state=tk.DISABLED)
        with open(f"output/{func}_summary.txt", "w") as f:
            f.write(summary)

# This function embeds plots into the tabs
# Input:
#       tab: The tab embedding into
#       fig: The figure that is getting embedded
def embedPlot(tab, fig):
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# This function builds the GUI and passes data
# Input:
#       processed: The processed data
#       refFronts: The reference pareto frontiers
def buildGui(processed, refFronts):
    root = tk.Tk()
    root.title("Multi-Objective Optimization Dashboard")
    notebook = ttk.Notebook(root)

    categories = ["Runtimes", "Pareto Frontiers", "IGD+ Graphs"]
    plotFuncs = {
        "Runtimes": [plotRuntime],
        "Pareto Frontiers": [plotBestFronts],
        "IGD+ Graphs": [plotIGDConvergence, plotAllIgd],
    }

    for category in categories:
        catTab = ttk.Notebook(notebook)
        notebook.add(catTab, text=category)

        for func in processed.keys():
            funcTab = ttk.Notebook(catTab)
            catTab.add(funcTab, text=func)

            for plotFunc in plotFuncs[category]:
                plotTab = ttk.Frame(funcTab)
                funcTab.add(plotTab, text=plotFunc.__name__.replace("plot_", "").replace("_", " ").title())

                if 'ref' in plotFunc.__code__.co_varnames:
                    fig = plotFunc(processed, refFronts, func)
                else:
                    fig = plotFunc(processed, func)
                embedPlot(plotTab, fig)

    notebook.pack(expand=1, fill="both")
    addSummaryTab(notebook, processed)
    root.mainloop()
buildGui(Processed, refFront)

