# Exploring Genetic Algorithm Techniques and Parameters for Solving Hardest 1-0 Knapsack Instances
### Ashwin Padmanabhan
#### September 2019
---

The section below \"Logic Review\" is a brief explanation of the logic
and assumptions used to implement the GA.

Logic Review:
=============

Overall Run Parameters: 
------------------------

-   **Representation**: A binary 1-D array of length equal to number of
    items available to place inside Knapsack. 0/1 representing item
    is/is not added in the Knapsack.

-   **Initial Population**: Start with 40 valid individuals chosen
    randomly. (Section 1.2)

-   **Progeny progression**: Two parents produce two children.

-   **Number of generations**: Most cases are run for 700 generations.
    This is chosen because this seems to be a good balance between
    having enough generations for convergence and computational cost for
    each generation.

Initialize Population:
----------------------

    def initializepopulation( instanceinfo, inpopulationsize, weight, 
                                knapsackcapacity, repmode = "binary"):
        if repmode == 'binary':
            initialpopulation =[]
            while( len(initialpopulation)< inpopulationsize ):
                generateindividual = bernoulli.rvs(0.01,
                                        size = int(instanceinfo[2]) )
                isfeasible = isFeasibleSolution(generateindividual,
                                        weight, knapsackcapacity)
                if ((isfeasible) and (listcompare(initialpopulation,
                                            generateindividual))):
                    initialpopulation.append(generateindividual)
            return(initialpopulation)

#### 

Initial population selection is done using random Bernoulli sampling.
**initializepopulation()** returns, \"initialpopulation\", a (Initial
population Size x len(genotype)) sized list. Population is initialized
using a Bernoulli trial value on each position of the genotype,
\"initial population size\" number of times. Within the loop (line 5),
line 8 checks if the genotype produced satisfies the capacity constraint
of the Knapsack and is appended into the population only if it does.
This is the constraint handling used here. Individuals (children,
mostly) are checked on whether they satisfy the Knapsack capacity
constraint and are added to the population only if they do. The
probability of the Bernoulli success is manually set to 0.01 using a
trial-error approach, since this produces best reasonable initial
individuals to get us started given the object weight and knapsack
capacity (especially for instance 2).

Selection:
----------

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

A tournament selection is implemented here. Line 5 above produces two
random indices between **\[0,len(population))** and a tournament takes
place to select the parent with a higher fitness value which then goes
on to be one of the parent.

Recombination and Mutation:
---------------------------

    #Recombination and Mutation. 
            for i in range(len(parentpair)):
                children = []
                generatecrossoverflag= bernoulli.rvs(crossoverrate, size = 1)
                if generatecrossoverflag ==1:
                    children= [np.concatenate((parentpair[i][0][0:crossoverpoint],parentpair[i][1][crossoverpoint:int(instanceinfo[2])]), axis =0),
                                    np.concatenate((parentpair[i][1][0:crossoverpoint], parentpair[i][0][crossoverpoint: int(instanceinfo[2])]), axis = 0)]
                else: 
                    children = [parentpair[i][0], parentpair[i][1]]
                #Mutation:
                children=mutate(children, mutationrate)
                for j in range(len(children)):
                    isfeasible = isFeasibleSolution(children[j], weights, knapsackcapacity)
                    if ((isfeasible) and listcompare(population, children[j])):
                        population.append(children[j])
                        population1.append([children[j], calcobj(children[j], values)])
                        presentindex = killindividual(population1)
                        population.pop(presentindex)
                        population1.pop(presentindex)

A mutation and crossover probability (or rate) is user assigned at the
beginning of the run. Crossover does a 1-pt crossover to produce the
genotype of the child. Once crossover is done, Mutation flips the value
of a genotype position to it's compliment. Also please note that
(following line 12) once children has been created they are appended
into the population only if the same genotype does not exist already in
the population (Boolean output function **listcompare()**) and if the
genotype produces a valid phenotype (Boolean output function
**isFeasible())** (Weight occupied by items in the Knapsack). If it is
infeasible, the child is not appended to the population.
**killindividual()** returns index to removes the same amount of
individuals as the number of children added to keep the overall
population constant.

#### Mutation:

Whenever a child is produced, mutation is done at every location of the
genotype with an equal probability as entered by the mutation rate. For
example, for a mutation rate of $10^{-5}$ every bit (position) of the
genotype is mutated equal Bernoulli success rate of $10^{-5}$. For
question 3, mutation does not take place every time a child is created.
It takes place with a probability given by the mutation rate (around
0.1), and whenever mutation is to be done, a random location from the
genotype is sampled and inverted. The code above calls the fitness
function **calcobj()** *only when* the child produced is not already
present in the existing population. The overall times the fitness
evalution function, calcobj(), is called: At the beginning when initial
population is defined and when a unique child is produced. This is the
reason why we do not need to make too many calls to the fitness
function.

Evolution:
----------

Below is a plot showing average fitness and optimal solution evolution
for the instance in file, knapPI\_13\_50\_1000.csv, running for 500
generations. Other evolution plots in submitted folder under \"Evolution
Plots\".

Impact of Mutation and Crossover:
=================================

**Study impact of the the balance between mutation and crossover (for
both uniform and single point) on algorithm run time (number of function
calls of the fitness function) and fitness.**

To study the balance of crossover and mutation rates, we will do a
contour plot for a range of cross-over and mutation probabilities,
plotting contours for both number of fitness function calls and average
fitness. Contour plots for the six instances given are shown below. The
GA is run for given set of repetitions (5 here, fig. 4 being an
exception, which was run twice) to calculate the average fitness value
at each of the crossover-mutation rate combinations.

\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_11_100_1000.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_11_100_1000_avgfitness.png}}
\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_13_50_1000_functioncalls.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_13_50_1000_avgfitness2.png}}
\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_13_200_1000.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_13_200_1000_fitness.png}}
\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_14_50_1000_functioncalls_4.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_14_50_1000_fitness.png}}
\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_15_50_1000.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_15_50_1000_fitness.png}}
\centering    
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_16_50_1000_3.png}}
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/MC_balance_knapPI_16_50_1000_fitness.png}}
Comparing the contour plots, it appears that a cross-over probability of
0.85 and a mutation probability of 0.000018 seems to not only produce
less number of function calls but provides a decent average optimal
solution.

Comparison to Random Sampling: 
===============================

**Compare the performance of the "best" parameters from part (a) to a
random sampling approach for different population sizes, ensuring a fair
comparison (equal total number of fitness function calls).**

\vspace{2mm}
I have scripted my program is such a way that the number of function
calls decides the population size for the Random Sampling (RS). The best
solution of the RS data was compared with the resulting best solution of
the GA. This comparison was repeated 20 times for 2 problem instances,
**\"knapPI\_13\_50\_1000.csv\"** and **\"knapPI\_14\_50\_1000.csv\"**.
In both cases GA performs significantly better than RS evidenced by the
line graph and the box plot below. The number of fitness calls made at
each run ranged from 250-400, which represents the population of the
randomly sampled data at those runs.

\centering    
\subfigure[For instance KnapPI14\_50\_1000]{\label{fig:MCB}\includegraphics[width=80mm]{images/GAvRSknap14_50_1000.png}}
\subfigure[For instance KnapPI13\_50\_1000 calls]{\label{fig:MCB}\includegraphics[width=80mm]{images/ga_rs_maxfitness_instance1.png}}
\centering    
\subfigure[For instance KnapPI14\_50\_1000]{\label{fig:MCB}\includegraphics[width=80mm]{images/GAvRS_Box_knap14_50_1000.png}}
\subfigure[For instance KnapPI13\_50\_1000 calls]{\label{fig:MCB}\includegraphics[width=80mm]{images/GAvRS_knap_13_50_100.png}}
\newpage
New mutation method:
====================

I am modifying the way mutation is being done. Earlier, a Bernoulli
trial was done for each position of the genotype for the child produced.
Instead, I think it is cleaner to first ask question \"Is this child a
mutant?\" which will give me a \"Yes\" with a mutation probability (say
around 0.01) and then I randomly select a position within the genotype
of the child and compliment the entry. I think this will be much more
scalable with increasing genotypic lengths. I ran the GA for 50
repetitions, for 500 iterations, each time, using both methods and noted
the average fitness of population in each case. Method 2 seems to do
much better than method 1, in terms of average fitness, however, at the
cost of higher fitness calls. (Histograms saved in the folder). I
performed a T test for the means of the two fitness/function calls
values as two independent samples assuming equal variance. I obtained a
p-value of 0.00417 and 0.12518 for fitness and function calls
respectively. It suggests that the difference in fitness is much more
significant that the difference in function calls. I would use this
method of mutation over the previous one.

The diagram below shows the distribution of function calls for the two
methods.

\centering
![[]{label=""}](images/comparemutation_functioncalls.png){width="85%"}

\centering    
\subfigure[Average Fitness]{\label{fig:MCB}\includegraphics[width=60mm]{images/M1_M2_Maxfitness.png}}
\subfigure[Number of fitness function calls]{\label{fig:MCB}\includegraphics[width=60mm]{images/M1_M2_Functioncall.png}}
