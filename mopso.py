import numpy as np
import matplotlib.pyplot as plt

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

def domination(particleA,particleB):
    return np.all(particleA <= particleB) and np.any(particleA < particleB)

def calcPareto(Particles):
    pareto = []
    N = Particles.shape[0]
    for i in range(N):
        parti = Particles[i]
        isDom = False

        for j in range(N):
            if i == j:
                continue
            if domination(Particles[j], parti):
                isDom = True
                break
        if not isDom:
            pareto.append(i)
    return pareto

# The driver for the MOPSO
def runMopso(testFunc, bounds, decision, objective, numGen=100, numParticle=100):
    # Initialize the population positions and velocities
    particles = np.array([[np.random.uniform(low,high) for (low,high) in bounds] for _ in range(numParticle)])
    print(particles)
    boundsFrac = (np.abs(bounds[0][0]) + np.abs(bounds[0][1]))*0.1
    velocities = np.random.uniform(-boundsFrac, boundsFrac, size=(numParticle,decision))
    
    # Initialize the personal best solution and fitness of each particle
    persDomSol = particles.copy()
    persDomFit = np.array([testFunc(x) for x in particles])

    # Initialize the global best solutions
    globDomSol = persDomSol.copy()
    globDomFit = persDomFit.copy()

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
        print(gener)
        fitnesses = np.array([testFunc(x) for x in particles])

        for i in range(numParticle):
            if(domination(fitnesses[i], persDomFit[i])):
                persDomSol[i] = particles[i]
                persDomFit[i] = fitnesses[i]
        
        # Calc up some global bests and archive them
        allFitness = np.vstack((globDomFit, fitnesses))
        print(len(allFitness))
        allParticle = np.vstack((globDomSol,particles))
        print(len(allParticle))
        paretoIndice = calcPareto(allFitness)
        paretoPos = allParticle[paretoIndice]
        paretoFit = allFitness[paretoIndice]
        print("CHECK LINE")

        # Random best position for example
        # Crowding REPLACEMENT

        if len(paretoIndice) > maxArchive:
            crowd1 = computeCrowdDist(paretoFit)
            topPar = np.argsort(-crowd1)[:maxArchive]
            paretoPos = paretoPos[topPar]
            paretoFit = paretoFit[topPar]
        globDomPos = paretoPos
        globDomFit = paretoFit
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
    for pos, fit in zip(globDomSol, globDomFit):
        print(f"x = {pos}, f = {fit}")
    
    plt.scatter(globDomFit[:, 0], globDomFit[:, 1])
    plt.show()



