import numpy as np
import matplotlib.pyplot as plt
import time

# Takes in the constraint scores and aggregate the pair based on whether
# The constraint is violated (High score) or not (0 for negative)
# Input:
#        con: An a tuple that contains the constraint scores for one particle
# Return:
#        The summed value of the constraint scores
def constraintViolate(con):
    return sum(max(0,coni) for coni in con)

# MOGA has NSGA-II, which is Godly. Needs Crowding to be fair
# This function, as a result, takes the fitnesses and scores them based on
# Distance to other particle fitnesses
# Input:
#        Particles: An array of fitness Values
# Output:
#        distance: The aggregate distance scores for each particle
def computeCrowdDist(Particles):
    """
    Compute the crowding distance for each point in F (N x M).
    """
    N, M = Particles.shape
    distance = np.zeros(N)

    for m in range(M):
        sortId = np.argsort(Particles[:, m])
        fMin = Particles[sortId[0], m]
        fMax = Particles[sortId[-1], m]

        distance[sortId[0]] = np.inf
        distance[sortId[-1]] = np.inf

        if fMax == fMin:
            continue

        for i in range(1, N - 1):
            prev = Particles[sortId[i - 1], m]
            nex = Particles[sortId[i + 1], m]
            distance[sortId[i]] += (nex - prev) / (fMax - fMin)

    return distance

# Takes in two particle fitnesses and their constraint scores to determine domination
# Feasibility is accounted for here
# Input:
#       particleA: The fitness vector for particle A
#       particleB: The fitness vector for particle B
#       conA: The constraint scores for particle A
#       conB: The constraint scores for particle B
# Return:
#       True or False: Based on whether A dominates B
def domination(particleA,particleB, conA, conB):
    vioA = constraintViolate(conA)
    vioB = constraintViolate(conB)
    if vioA == 0 and vioB == 0:
        return np.all(particleA <= particleB) and np.any(particleA < particleB)
    elif vioA == 0:
        return True
    elif vioB == 0:
        return False
    else:
        return vioA < vioB

# This function calculates the pareto frontier by storing indices
# Input:
#        Particles: An array containing the particle fitness scores
#        Constraints: An array containing the particle constraint scores
# Output:
#        pareto: An array of indices for the pareto frontier
def calcPareto(Particles, Constraints):
    pareto = []
    N = Particles.shape[0]
    for i in range(N):
        parti = Particles[i]
        coni = Constraints[i]
        isDom = False

        for j in range(N):
            if i == j:
                continue
            if domination(Particles[j], parti, Constraints[j], coni):
                isDom = True
                break
        if not isDom:
            pareto.append(i)
    return pareto

# The driver for the MOPSO
# Has the following features and modifications: Crowding, Global Archive Pruning, random leader selection per particle
# Boundary clipping on extreme edges.
# Input:
#       testFunc: The function that determines the fitness of a particle 
#       bounds: The bounds array containing the (L,H) for each particle
#       decision: The number of decision variables per particle
#       objective: The number of object variables per particle
#       conNum: The number of constraint values per particle
#       conFun: The constraint function if applicable. Default is None
#       numGen: The number of generations the MOPSO will run. Default is 100
#       numParticle: The number of particles in the swarm. Default is 100
# Output:
#       A list containing: results (solution and fitness), runtime, and history
def runMopso(testFunc, bounds, decision, objective, conNum = 0, conFun = None, numGen=100, numParticle=100):
    # Initialize the population positions and velocities
    particles = np.array([[np.random.uniform(low,high) for (low,high) in bounds] for _ in range(numParticle)])
    boundsFrac = (np.abs(bounds[0][0]) + np.abs(bounds[0][1]))*0.1
    velocities = np.random.uniform(-boundsFrac, boundsFrac, size=(numParticle,decision))

    # If conFun IS None, we need an arbitary function so we can navigate as normal
    if conFun is None:
        def conFun(x):
            return np.zeros(1)
    
    # Initialize the personal best solution and fitness of each particle
    persDomSol = particles.copy()
    persDomFit = np.array([testFunc(x) for x in particles])
    persDomCon = np.array([conFun(x) for x in particles])

    # Initialize the global best solutions
    globDomSol = persDomSol.copy()
    globDomFit = persDomFit.copy()
    globDomCon = persDomCon.copy()

    # I don't necessarily care about the solution set history, I just want the fitness history
    FitArchive = []

    # Initialize w, c1, c2, and max archive
    w = 0.4
    c1 = 1.5
    c2 = 1.5
    maxArchive = 200
    # Slip bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    start = time.time()
    # Begin Optimization
    for gener in range(numGen):
        fitnesses = np.array([testFunc(x) for x in particles])
        constraints = np.array([conFun(x) for x in particles])

        for i in range(numParticle):
            if(domination(fitnesses[i], persDomFit[i], constraints[i], persDomCon[i])):
                persDomSol[i] = particles[i]
                persDomFit[i] = fitnesses[i]
                persDomCon[i] = constraints[i]
        
        # Calc up some global bests and archive them
        allFitness = np.vstack((globDomFit, fitnesses))
        allParticle = np.vstack((globDomSol, particles))
        allCon = np.vstack((globDomCon, constraints))
        paretoIndice = calcPareto(allFitness, allCon)
        paretoPos = allParticle[paretoIndice]
        paretoFit = allFitness[paretoIndice]
        paretoCon = allCon[paretoIndice]

        if len(paretoIndice) > maxArchive:
            crowd1 = computeCrowdDist(paretoFit)
            topPar = np.argsort(-crowd1)[:maxArchive]
            paretoPos = paretoPos[topPar]
            paretoFit = paretoFit[topPar]
            paretoCon = paretoCon[topPar]
        globDomSol = paretoPos
        globDomFit = paretoFit
        globDomCon = paretoCon
        crowd = computeCrowdDist(globDomFit)
        FitArchive.append((gener, globDomFit))
        finiteCrowd = crowd[np.isfinite(crowd)]
        if len(finiteCrowd) == 0:
            maxim = 1.0
        else:
            maxim = np.max(finiteCrowd)
        crowd[np.isinf(crowd)] = maxim * 2
        crowd += 1e-6
        
        r1 = np.random.rand(numParticle, decision)
        r2 = np.random.rand(numParticle, decision)
        
        crowdSum = np.sum(crowd)
        if crowdSum == 0.0:
            prob = np.ones(len(crowd))/len(crowd)
        else:
            probOnCrowd = crowd / crowdSum
        for i in range(numParticle):
            leaderID = np.random.choice(len(globDomSol), p=probOnCrowd)
            leader = globDomSol[leaderID]
            velocities[i] = (w * velocities[i] + c1 * r1[i] * (persDomSol[i] - particles[i]) +
            c2 * r2[i] * (leader - particles[i]))
        particles += velocities
        for i in range(len(particles)):
            particles[i] = np.clip(particles[i], lower, upper)
    end = time.time()
    runtime = end-start
    result = (globDomSol, globDomFit)
    return [result, runtime, FitArchive]
