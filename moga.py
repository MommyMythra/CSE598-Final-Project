import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2

# Pymoo function wrapper. It will take in any test function and bounds and allow it to be properly run by pymoo. 
class TestWrap(Problem):
    def __init__(self, userFunc, bounds, decision, constrFunc = None):
        self.userFunc = userFunc
        self.constrFunc = constrFunc
        self.lower, self.upper = zip(*bounds)
        nVar = decision
        test = np.zeros(nVar)
        testOutput = userFunc(test)
        nObj = len(testOutput)
        nConstr = len(constrFunc(test)) if constrFunc is not None else 0

        super().__init__(n_var=nVar,
                         n_obj=nObj,
                         n_constr=nConstr,
                         xl=np.array(self.lower),
                         xu=np.array(self.upper))

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.userFunc(x) for x in X])
        if self.constrFunc is not None:
            out["G"] = np.array([self.constrFunc(x) for x in X])

# This function runs the pymoo instance. It defaults to the provided NSGA-II, 100 generations, and 100 pop size. 
# It does collect history for graphs later
def runMoga(userFunc, bounds, decision, confunc = None, nGen=100, popSize=100):
    problem = TestWrap(userFunc, bounds, decision, confunc)
    algorithm = NSGA2(pop_size=popSize)
    
    result = minimize(problem,
                      algorithm,
                      termination=get_termination("n_gen", nGen),
                      save_history=True)
    
    return result
