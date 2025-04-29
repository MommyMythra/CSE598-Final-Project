import numpy as np
import matplotlib.pyplot as plt

def constraintViolate(con):
    return sum(max(0,coni) for coni in con)

# MOGA has NSGA-II, which is Godly. Needs Crowding to be fair
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

    # Initialize w, c1, c2, Xo, and pert
    # This follows Optimized PSO weightings from Fuzhang Zhao
    #Xo = (np.sqrt(5)-1)/2
    #c1 = (np.sqrt(5)-1)/2
    #c2 = (3-np.sqrt(5))/2
    #w = 1-Xo
    #pert = 2 + np.sqrt(5)
    w = 0.4
    c1 = 1.5
    c2 = 1.5
    maxArchive = 250
    # Slip bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

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

        # Random best position for example
        # Crowding REPLACEMENT

        if len(paretoIndice) > maxArchive:
            crowd1 = computeCrowdDist(paretoFit)
            topPar = np.argsort(-crowd1)[:maxArchive]
            paretoPos = paretoPos[topPar]
            paretoFit = paretoFit[topPar]
            paretoCon = paretoCon[topPar]
        globDomPos = paretoPos
        globDomFit = paretoFit
        globDomCon = paretoCon
        crowd = computeCrowdDist(globDomFit)
        TheBest = globDomSol[np.argmax(crowd)]
        
        r1 = np.random.rand(numParticle, decision)
        r2 = np.random.rand(numParticle, decision)

        # Calculate the velocities using modified equations from Fuzhang Zhao
        velocities = (
                w * velocities +
                #Xo * c1  * r1 * (persDomSol - particles) +
                #Xo * c2  * r2 * (TheBest - particles)
                c1 * r1 * (persDomSol - particles) +
                c2 * r2 * (TheBest - particles)
                )
        
        # Calc the final positions and BIND them
        particles += velocities
        for i in range(len(particles)):
            particles[i] = np.clip(particles[i], lower, upper)

    print("Some final results?: ")
    for pos, fit, con in zip(globDomSol, globDomFit, globDomCon):
        print(f"x = {pos}, f = {fit}, con = {con}")
    print(len(globDomSol))
