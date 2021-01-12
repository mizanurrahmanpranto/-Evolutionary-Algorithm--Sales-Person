# Evolutionary-Algorithm-Travel Sales Person

What is Genetic Algorithm(GA?
Genetic algorithms are randomized search algorithms that have been developed in an effort to imitate the mechanics of natural selection and natural genetics. Genetic algorithms operate on string structures, like biological structures, which are evolving in time according to the rule of survival of the fittest by using a randomized yet structured information exchange. Thus, in every generation, a new set of strings is created, using parts of the fittest members of the old set. The main characteristics of a genetic algorithm are as follows:

(1) The genetic algorithm works with a coding of the parameter set, not the parameters themselves.
(2) The genetic algorithm initiates its search from a population of points, not a single point.
(3) The genetic algorithm uses payoff information, not derivatives.
(4) The genetic algorithm uses probabilistic transition rules, not deterministic ones.

At first, the coding to be used must be defined. Then using a random process, an initial population of strings is created. Next, a set of operators is used to take this initial population to generate successive populations, which hopefully improve with time. The main operators of the genetic algorithms are reproduction, crossover, and mutation.
Reproduction is a process based on the objective function (fitness function) of each string. This objective function identifies how “good” a string is. Thus, strings with higher fitness value have bigger probability of contributing offsprings to the next generation.
Crossover is a process in which members of the last population are mated at random in the mating pool. So, a pair of offsprings is generated, combining elements from two parents (members), which hopefully have improved fitness values. Mutation is the occasional (with small probability) random alteration of the value of a string position. In fact, mutation is a process of random walk through the coded parameter space. Its purpose is to ensure that important information contained within strings may not be lost prematurely.
The implementation of GAs to the problem of optimization of operation and site-scale energy use in production plants is envisaged to be carried along the following lines.
First, using the pinch design method, the problem requirements and constraints will be defined. Next, the region of the pinch will be identified, and the essential matches will be made. Then, the problem will be coded by generating the appropriate strings. These strings will contain the general features and the parameters, which affect the problem. Each string will represent a possible network configuration. For each one of the strings, the objective function will be calculated, that is, the value of energy and utility usage and the number of units need to be used in the represented network configuration, that is, the total capital and operating cost. Initially, a starting population of strings will be created using a random procedure. The three main operators of the genetic algorithms will be performed to improve the value of the objective function, namely, to create network configurations with minimal total capital and operating cost. To the final population of strings/possible networks, advanced techniques will be applied for further improvement.
Genetic algorithm is a kind of stochastic algorithm based on the theory of probability. In application this method to a stagewise superstructure model, the search process is determined by stochastic strategy. The global optimal solution for the synthesis of heat exchange networks can be obtained at certain probability. The search process begins with a set of initial stochastic solutions, which is called “population.” Each solution is called “chromosome,” the chromosome is composed of “gene,” and the “gene” stands for the optimal variables of heat exchange networks, for example, the mass flowrates of cold streams and hot streams.
There are two kinds of calculation operation in the genetic algorithm: genetic operation and evolution operation. The genetic operation adopts the transferring principle of probability, selects some good chromosomes to propagate at certain probability, and lets the other inferior chromosomes to die; thus, the search direction will be guided to the most promising region. With a stochastic search technique, they can explore different regions of the search space simultaneously and hence are less prone to terminate in local minimum. The strength of the genetic algorithm is the exploration of different regions of the search space in relatively short computation time. Furthermore, multiple and complex objectives can easily be included. But genetic algorithm provides only a general framework for solving complex optimization problem. The genetic operators are often problem-dependent and are of critical importance for successful use in practical problem. Specifically, to the synthesis problem of heat exchanger networks with multistream heat exchangers, an approach for initial network generation, heat load determination of a match within superstructure should be given. Some operators such as crossover operator, mutation operator, orthogonal crossover, and effective crowding operators are appropriately designed to adapt to the synthesis problem. Another difficulty for genetic algorithm application is the treatment of constraints. During the genetic evolution, an individual of the population may turn into infeasible solution after manipulated by genetic operators, which will lead to failure to find a feasible solution during evolution, especially for the optimization problem with strict constraints. Hence, some strategy should be contrived for constraints guarantee in genetic computation.
Find the Shortest path?
Genetic algorithm is used for analyzing business problems mostly applied to find solution for business challenges. Genetic algorithm generates many solutions to a single problem each one with different performance some are better than other in performance.
Finding shortest path has many applications in different fields. The purpose of this paper is to find the business problem related to supermarkets and give the solution to this problem using genetic algorithm. In this paper we chose supermarkets in different location and we want to find the shortest path for the manager to visit among these supermarkets with shortest distance, therefore in genetic algorithm the evaluation function finds all the relevant variables of Geno type and then evaluates fitness function on these Geno variables using crossover and mutation techniques for sorting out relevant values. In GA the fitness function is the total time elapsed for the target to achieve their goal. We use the fitness function on population data with respect to crossover and mutation function for training data sets. The main problems in Business, science and engineering are to find the short path in different activities like visiting different places or transferring some data in minimum time.
The genetic algorithm provides efficient method to provide optimization of these problems. To solve the supermarket manger traveling problem we encode the data sets of our problem using GA and initialize all the variables. The estimation values of the locations are set as a parameter to genetic algorithm and find the best method by using the MDL technique (Minimum Description length). Genetic algorithms are very efficient for selection and genetics of natural values.




Fitness and DNA
The way of encoding DNA this time is different. We can try to have an ID for each city, and the order of the cities experienced is sorted by ID. For example, if a merchant passes through 3 cities, we have
0-1-2
0-2-1
1-0-2
1-2-0
2-0-1
2-1-0
These 6 arrangements. Each arrangement can be regarded as a DNA sequence. The way to generate this DNA sequence with numpy is very simple.

>>> np.random.permutation(3) 
# array([1, 2, 0])


Calculation of fitness, we as long as the DNA in these cities together into line, calculate the total path length, depending on the length, we set the rules, the better the shorter total path, following fitness0it used to calculate the fitness friends. because the shorter the path we choose the asking price significantly, so here I used fitness1this way.


class GA:
def select(self, fitness):

def crossover(self, parent, pop):

def mutate(self, child):

def evolve(self):
Algorithm above these functions in this section have described in detail so will not be described in detail. You can also go to my github see all the code , but we should note that in crossoverand mutatehave a little bit of time is not the same , Because we can't change the way points at will. For example, if we follow the usual way crossover, it may be like this:

p1=[0,1,2,3] (father)
p2=[3,2,1,0] (mom)
cp=[m,b,m,b] (Intersection, m: mom, b: dad)
c1=[3,1,1,3] (child)
Then this c1to go through two city 3, 1 two cities, but without 2, 0. Obviously not. So we are crossoverand mutationhave to put it another way. One possible way is. Also above example .
p1=[0,1,2,3] (father)
cp=[_,b,_,b] (Choose the point from Dad)
c1=[1,3,_,_] (First fill in the father's dot in front of the child)
At this time, in addition to 1, as well as 3. 0, 2 father from the two cities, on the order of 0,2 arranged in the mother's DNA sequence. Is p2=[3,2,1,0]0, 2 is two cities in the prior p2 2, Then there is 0. So we fill in the DNA of the child in this order.

c1=[1,3,2,0]
In this way, we will succeed in avoiding crossoverproblems arising: access by multiple problems of the city written in Python is simple.
If np.random.rand() < self.cross_rate: i_ = np.random.randint(0, self.pop_size, size=1) # select another individual from pop 
cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool) # choose crossover points 
keep_city = parent[~cross_points] # find the city number 
swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]  
parent[:] = np.concatenate((keep_city, swap_city))

At mutatethe time, also found two different DNA points, then swap the two points just fine.

for point in range(self.DNA_size): 
if np.random.rand() < self.mutate_rate: 
swap_point = np.random.randint(0, self.DNA_size) 
swapA, swapB = child[point], child[swap_point] child[point], child[swap_point] = swapB, swapA

The final loop main frame remains the same, as simple as the following.
ga = GA(...) for generation in range(N_GENERATIONS): fitness = ga.get_fitness() ga.evolve(fitness)

Code Details
N_CITIES = 20 # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500

ef translateDNA(self, DNA, city_position): # get cities' coord in order
line_x = np.empty_like(DNA, dtype=np.float64)
line_y = np.empty_like(DNA, dtype=np.float64)
for i, d in enumerate(DNA):
city_coord = city_position[d]
line_x[i, :] = city_coord[:, 0]
line_y[i, :] = city_coord[:, 1]
return line_x, line_y

def crossover(self, parent, pop):
if np.random.rand() < self.cross_rate:
i_ = np.random.randint(0, self.pop_size, size=1) # select another individual from pop
cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool) # choose crossover points
keep_city = parent[~cross_points] # find the city number
swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
parent[:] = np.concatenate((keep_city, swap_city))
return parent

def evolve(self, fitness):
pop = self.select(fitness)
pop_copy = pop.copy()
for parent in pop: # for every parent
child = self.crossover(parent, pop_copy)
child = self.mutate(child)
parent[:] = child
self.pop = pop
