#%% Main file run. 
#Importing py packages and "func.py"
import csv
import pickle
from scipy.stats import bernoulli
import numpy as np
import random
from statistics import mean
import matplotlib.pyplot as plt
import func
import importlib
import plotly.express as px
import pandas as pd
from scipy import stats
importlib.reload(func)

#%%#
# 
# ============== SET :: System variables ============== ============== ============== ============== ============== 

#initial population size.
inpopulationsize = 20
#Number of parent "pairs" in each gen. 
numberparents = 2 
#placeholder. Crossover pts randomly sampling a number between (0, len(genotype))
crossoverpoint= 15 
#knapsack capacity. HARD-CODED
knapsackcapacity = 997
#mutation probability
mutationrate = 0.18
#crossover probability
crossoverrate = 0.85
#Number of generations to be run. 
numgen = 1000

#Contour plot settings
#Crossover rate range
crrange = tuple((0.75,0.98))
#Mutation rate range
mrrange = tuple((0.000005, 0.00002))
repetitions = 3
#binsize for Countour plots
binsize = 3

murate = 0.1

#Filename: the two file names are essentially the same. instancefilename 
#is the file provieded. "datafilename" is modified filename with single 
#sub-instance and summary statistics (provided in the begining of instancefilename removed)
datafilename = "secondinstance.csv"
instancefilename = "knapPI_13_50_1000.csv"

#%% Function call to run the genetic algorithm. 
#Mutationmethod = "method1"/"method2"
#needplot = True to show plot
a= func.ga(inpopulationsize,numberparents, crossoverpoint, knapsackcapacity, murate
        , crossoverrate, numgen, datafilename, instancefilename ,needplot= True, mutationmethod= "method2")  

#%% Function call to generate contour plot.
# 
# measure = callcount/maxfitness
func.getcountour(inpopulationsize, numberparents, crossoverpoint, 
                    knapsackcapacity, numgen, datafilename, instancefilename, 
                    1, crrange, mrrange, "callcount", binsize) 

#%%New mutation method
a= func.ga(inpopulationsize,numberparents, crossoverpoint, knapsackcapacity, mutationrate
        ,crossoverrate, numgen, datafilename, instancefilename, needplot= True, mutationmethod= "method1")  

#%% New mutation method implementation. 

## ==============================COMPARITIVE STUDY OF THE TWO METHODS:==============================

## ============================== Run method 1. ## ==============================
repetitions1 = 20
res1 = func.runga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, 
            mutationrate, crossoverrate, numgen, datafilename, instancefilename, 
            repetitions1,  mmethod = "method1", savevar= False)

## ============================== Run Method 2. ## ==============================
#repetitions1 = 20
mrate = 0.1
res2 = func.runga(inpopulationsize, numberparents, crossoverpoint, knapsackcapacity, 
            mrate, crossoverrate, numgen, datafilename, instancefilename, 
            repetitions1,  mmethod = "method2", savevar= False)

#%% 
# ============================== ============================== ==============================
# ============================== ============================== ==============================

# ANALYSING and PLOTTING THE TWO METHODS

## ============================== CREATE HISTOGRAMS ## ==============================

## =========== Histogram measured againt fitness.  ===================
fitness1 = [i["maxfitness"] for i in res1]
fitness2 = [i["maxfitness"] for i in res2]
legend = ['Method1', 'Method2']
plt.hist([fitness1, fitness2], color=['orange', 'green'])
plt.xlabel('Max fitness')
plt.legend(legend)
plt.ylabel('Counts')
plt.title('Histogram of fitness comparing the Two Mutation methods')

## ============ Histogram measured againt function calls. ==================

funccall1  = [i["callcount"] for i in res1]
funccall2  = [i["callcount"] for i in res2]
plt.hist([funccall1, funccall2], color=['orange', 'green'])
plt.xlabel('Fitness function calls')
plt.legend(legend)
plt.ylabel('Counts')
plt.title('Histogram of fitness function calls comparing the Two Mutation methods')


## ============================== CREATE BOX-WHISKER PLOTS ## ==============================

## ## =========== BOX-Whisker plots measured againt fitness.  ===================
name1= ["Method1" for i in fitness1]
name2= ["Method2" for i in fitness2]

newfitness1 = [[fitness1[i], name1[i]] for i in range(len(fitness1))]
newfitness2 = [[fitness2[i], name2[i]] for i in range(len(fitness2))]

df_maxfitness = pd.DataFrame(newfitness1 + newfitness2)
df_maxfitness.columns= ["Maxfitness", "methods"]

fig = px.box(df_maxfitness, x = "methods",  y="Maxfitness")
fig.show()

## =========== BOX-Whisker plots measured againt function calls.  ===================
calls1 = [[funccall1[i], name1[i]] for i in range(len(funccall1))]
calls2 = [[funccall2[i], name2[i]] for i in range(len(funccall2))]

df_funccall = pd.DataFrame(calls1 + calls2)
df_funccall.columns= ["FuncCalls", "methods"]

fig = px.box(df_funccall, x = "methods",  y="FuncCalls")
fig.show()

stats.ttest_ind(fitness1,fitness2)
stats.ttest_ind(funccall1,funccall2)


#%% 
# ============================== ============================== ==============================
# ============================== ============================== ==============================

# Comparitive study of GA with Random Sampling approach. 

# ============================================================

repetitions2 = 10
rsmaxfitcollect = []
gamaxfitcollect= []

rsavgfitcollect = []
gaavgfitcollect= []

#Running GA and random sampling:: 
 
for i in range(repetitions2):
        #print(i)
        gaop= func.ga(inpopulationsize,numberparents, crossoverpoint, knapsackcapacity, mutationrate
                ,crossoverrate, numgen, datafilename, instancefilename, needplot= False, mutationmethod= "method2")  

        instanceinfo= func.getinstanceinfo(instancefilename)
        data = func.importfile(datafilename)
        values = np.array(func.getvalues(data))
        weights = np.array(func.getweight(data))

        rsop = func.initializepopulation(instanceinfo, gaop["callcount"], weights, knapsackcapacity)
        rsop1 = func.appendphenotype(rsop, values)

        rsbestsolution = func.getbestsolution(rsop1, values)
        rsmaxfitcollect.append(rsbestsolution[1])

        rsavgfitcollect.append(func.getpopulationfitness(rsop, values))
        gamaxfitcollect.append(gaop["maxfitness"])
        gaavgfitcollect.append(gaop["meanfitness"])

##Box plot 
## For fitness:
GAname= ["GA" for i in gamaxfitcollect]
RSname= ["RS" for i in rsmaxfitcollect]

GA = [[gamaxfitcollect[i], GAname[i]] for i in range(len(gamaxfitcollect))]
RS = [[rsmaxfitcollect[i], RSname[i]] for i in range(len(rsmaxfitcollect))]

dfdf = pd.DataFrame(GA + RS)
dfdf.columns= ["Maxfitness", "methods"]

fig = px.box(dfdf, x = "methods",  y="Maxfitness")
fig.show()
#%%
