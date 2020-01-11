#GA for Knapsack problem

# Run all functions
#%% 
import csv
from scipy.stats import bernoulli
import numpy as np
import random
from statistics import mean, median
import matplotlib.pyplot as plt
import pickle
import random 

#Function to read instance file. 
def importfile(filename):
    with open(filename) as filename:
        rows = csv.reader(filename)
        for row in enumerate(rows): 
            if row[0] == 0:
                data = []
                data.append(row)
            else: 
                data.append(row)
    return data
#Function to extract information from filename. 
def getinstanceinfo(filename):
    a = filename.split("_")
    return a
#Fitness calculation function call for a given genotype. 
def calcobj(genotype, values):
    val= 0
    for i in range(len(genotype)):
            val= val + (genotype[i]*values[i])
    return(val)
#Weight calculation function call. Gives weight filled in knapsack 
def calcweight(genotype, weight):
    val= 0
    for i in range(len(genotype)):
            val= val + (genotype[i]*weight[i])
    return(val)
#Selection function. population1 is [[population], fitness].
def selection(population, population1, values, mode= "tournament"):
    if mode == "tournament": 
        parents = []
        for _ in range(2):
            candidateparent = [random.randrange(0,len(population)) 
                                for _ in range(2)] 
            parent = np.argmax(
                [population1[candidateparent[0]][1], 
                    population1[candidateparent[1]][1]])
            parent = population[candidateparent[parent]]
            parents.append(parent)
        return(parents)
#Boolean function for constraint checking. 
def isFeasibleSolution(genotype, weight, knapsackcapacity):
    val = calcweight(genotype, weight)
    if val > knapsackcapacity:
        return 0
    else: 
        return 1
#Boolean function to check if "generateindividual" exists in "initialpopulation"
def listcompare(initialpopulation, generateindividual):
    if initialpopulation == []:
        return True
    else:   
        for i in range(len(initialpopulation)):
            if np.all(initialpopulation[i] == generateindividual):
                return(False)
    return(True)
#Function to initialize population. 
def initializepopulation( instanceinfo, inpopulationsize, weight, 
                            knapsackcapacity, repmode = "binary"):
    if repmode == 'binary':
        initialpopulation =[]
        while( len(initialpopulation)< inpopulationsize ):
            #generate individual. 
            generateindividual = bernoulli.rvs(0.01, size = int(instanceinfo[2]) )
            #is generated individual valid?
            isfeasible = isFeasibleSolution(generateindividual, weight, knapsackcapacity)
            #if valid and does not already exist in population, add to population.
            if ((isfeasible) and (listcompare(initialpopulation, generateindividual))):
                initialpopulation.append(generateindividual)
        return(initialpopulation)
#performs mutation on each child and get genotype value. Reason why mutation
#ratio must be small. 
def mutate(children, mutationrate):
    for i in range(len(children)):
        for j in range(len(children[i])):
            generatemutateflag= bernoulli.rvs(mutationrate, size = 1)
            if generatemutateflag[0] == 1:
                if children[i][j] == 0:
                    children[i][j] = 1
                else: 
                   children[i][j] = 0
    return children 

def newmutate(children, murate, instanceinfo):
    for i in range(len(children)):
        genflag= bernoulli.rvs(murate, size = 1)
        if genflag == True:
            index = random.randrange(0, int(instanceinfo[2]), 1)
            if children[i][index] == 1:
                children[i][index] = 0
            else:
                children[i][index] = 1
    return children

#Function to calculate overall mean fitness of population. 
def getpopulationfitness(population, value):
    fitness= []
    fitness = [ population[i][1] for i in range(len(population))]
    return mean(fitness)
#Function to calculate best solution in a population 
def getbestsolution(population1, value):
    fitnessmax = -100
    bestgeno = []
    for i in range(len(population1)):
        presentfit = population1[i][1]
        if presentfit > fitnessmax:
            bestgeno = population1[i][0]
            #print("Value of best genotype is: ", presentfit )
            fitnessmax = presentfit
    return [bestgeno, fitnessmax]
#function to initially append population1 with phenotypic value during 
#initializepop()
def appendphenotype(population, value):
    population = [ [population[i]] + [calcobj(population[i], value)] 
                        for i in range(len(population)) if len([population[i]]) == 1]
    return population
#function to remove individual with lowest phenotypic value in case child added.
def killindividual(population): 
    minpheno = 10**10
    presentindex = -1
    for i in range(len(population)):
        if minpheno>population[i][1]:
            minpheno = population[i][1]
            presentindex = i
    return presentindex
#get values of values and weights from data
getvalues= lambda data: [int(d[1][1]) for d in data]
getweight = lambda data: [int(d[1][2]) for d in data]

#%% Function to perform GA.
def ga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, 
        mutationrate, crossoverrate, numgen, datafilename, instancefilename, 
        repmode = "binary", selectionmode = "tournament", needplot = False, mutationmethod = "method1"):
    #Set up GA
    objcallcount = 0
    #----------------------
    #Get values: 
    data = importfile(datafilename)
    values = np.array(getvalues(data))
    weights = np.array(getweight(data))
    #----------------------
    #Define the initial population:
    initialpopulation, population, population1 = [], [], []
    instanceinfo= getinstanceinfo(instancefilename)
    initialpopulation = initializepopulation(instanceinfo, inpopulationsize , weights, knapsackcapacity, repmode = "binary")
    population = initialpopulation
    population1 = appendphenotype(population, values)
    objcallcount = objcallcount + len(population1)
    fitness= []
    bestfitnessevolution = []
    #print(getpopulationfitness(population1, values))
    #Fitness tracts fitness with generation
    fitness.append(getpopulationfitness(population1, values))
    #optimalinfo gives best genotype and best solution in present gen.
    optimalinfo = getbestsolution(population1, values)
    bestfitnessevolution.append(optimalinfo[1])
    convergenceiter = 0
    # Run GA
    #----------------------
    for ii in range(numgen):
        #Selection: 
        parentpair = [ selection(population, population1, values, "tournament") for _ in range(numberparents)] #List of list with each parent. 
    #----------------------
        #Recombination and Mutation. 
        for i in range(len(parentpair)):
            children = []
            generatecrossoverflag= bernoulli.rvs(crossoverrate, size = 1)
            if generatecrossoverflag[0] ==1:
                crossoverpoint = random.randrange(0, int(instanceinfo[2]), 1)
                children= [np.concatenate((parentpair[i][0][0:crossoverpoint],parentpair[i][1][crossoverpoint:int(instanceinfo[2])]), axis =0),
                                np.concatenate((parentpair[i][1][0:crossoverpoint], parentpair[i][0][crossoverpoint: int(instanceinfo[2])]), axis = 0)]
            else: 
                children = [parentpair[i][0], parentpair[i][1]]
            #Mutation:
            if mutationmethod == "method1":
                children=mutate(children, mutationrate)
            else: 
                children=newmutate(children, mutationrate, instanceinfo)
            for j in range(len(children)):
                isfeasible = isFeasibleSolution(children[j], weights, knapsackcapacity)
                if ((isfeasible) and listcompare(population, children[j])):
                    population.append(children[j])                        
                    population1.append([children[j], calcobj(children[j], values)])
                    objcallcount+=1
                    presentindex = killindividual(population1)
                    population.pop(presentindex)
                    population1.pop(presentindex)
        optimalinfo = getbestsolution(population1, values)
        bestfitnessevolution.append(optimalinfo[1])
        if bestfitnessevolution[-1] != bestfitnessevolution[-2]:
            convergenceiter = ii
        fitness.append(getpopulationfitness(population1, values))
    if needplot == True:
        plot(numgen, fitness, bestfitnessevolution, weights, values, instancefilename)
    #----------------------
    #Get best solution:
    bestgeno = getbestsolution(population1, values)
    print("CR= ", crossoverrate)
    print("MR= ", mutationrate)

    print("Value for best genotype is: ", bestgeno[1])
    print("Objective call count is: ", objcallcount)
    
    retdict = {
            "maxfitness" : bestgeno[1], 
            #"best_genotype" : bestgeno[0], 
            #"weightoccupied" : calcweight(bestgeno[0], weights), 
            #"solution" : bestfitnessevolution[-1],
            "callcount": objcallcount, 
            #"convergenceiternum": convergenceiter, 
            #"mutationrate": mutationrate, 
            #"crossoverrate": crossoverrate, 
            "meanfitness": fitness[-1], 
            #"solutionevol": bestfitnessevolution
            #"bestsolutionevolution": bestfitnessevolution,
            #"medianfitness": median(fitness)
            }
    return retdict

#Run ga for a given MR, CR, "repetition" number of times and return as a list of dict of len(repetition)
def runga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, 
            mutationrate, crossoverrate, numgen, datafilename, instancefilename, 
            repetitions,  mmethod = "method1", savevar= False): 
    res = []
    collect = []
    for _ in range(repetitions):
        res = ga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, mutationrate, crossoverrate, numgen, datafilename, instancefilename, mutationmethod= mmethod)
        collect.append(res)
    if savevar == True: 
        with open('runGA_dump.pkl', 'wb') as f:
            a = pickle.dump(collect, f) 
    return collect

def plot(numgen,fitness,bestfitnessevolution, weights, values, instancefilename):
    plt.plot(range(0,numgen+1), fitness)
    plt.title(instancefilename[0:-4])
    plt.xlabel("Generation Number")
    plt.ylabel("Average Fitness")
    plt.show()

    plt.plot(range(0,numgen+1), bestfitnessevolution)
    plt.title(instancefilename[0:-4])
    plt.xlabel("Generation Number")
    plt.ylabel("Optimal value")
    plt.show()

#function that "CALLS FUNCTION GA" and plots contours. 
def getcountour(inpopulationsize, numberparents, crossoverpoint, 
                knapsackcapacity, numgen, datafilename, instancefilename, 
                repetitions, crrange, mrrange, measure, binsize):
    #Crossoverrate in x axis:
    xlist = np.linspace(crrange[0], crrange[1], binsize)
    # Mutation rate in y axis:
    ylist = np.linspace(mrrange[0], mrrange[1], binsize)
    X, Y = np.meshgrid(xlist, ylist)

    if measure == "callcount":
        fullz = []
        for i in xlist: 
            a1 = []
            for j in ylist:
                res= ga(inpopulationsize,numberparents, crossoverpoint, knapsackcapacity, j
                        , i, numgen, datafilename, instancefilename, needplot= False)  
                a1.append(res["callcount"])
            fullz.append(a1)

    if measure == "maxfitness":
        fullz = []
        for i in xlist: 
            presentz = []
            for j in ylist:
                a = []
                for _ in range(repetitions):
                    res = ga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, j, i, numgen, datafilename, instancefilename)
                    a.append(res["maxfitness"])
                presentz.append(mean(a))
            fullz.append(presentz)

    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    cp = ax.contour(X, Y, fullz)
    ax.clabel(cp, inline=True, 
            fontsize=10)
    ax.set_title('Contour_'+instancefilename[0:-4])
    ax.set_xlabel('Crossover Rate')
    ax.set_ylabel('Mutation Rate')
    plt.show()

#%%
