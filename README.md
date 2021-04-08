# ELECTRE-Tree

Draft: https://arxiv.org/abs/2007.10047             Article: https://www.emerald.com/insight/content/doi/10.1108/DTA-10-2020-0256/full/html

**Try it online**: ([ Colab Demo ](https://colab.research.google.com/drive/1eAJ89u2gqMxp4pIfUeBHQlJqrb64V7_D?usp=sharing))

ELECTRE-Tree Algorithm to infer the ELECTRE Tri-B method parameters. The function returns: 1) A list of optimized sub-models that can be used to vote the allocation of alternatives (assign to a class) or can infer the ELECTRE Tri-B parameters using the average.


<br/> __*"tree_electre_tri_b" arguments*__<br/>
 
* dataset = A numpy array where the rows are the alternatives and columns are the criteria. 

* target_assignment = Optional argument. A list of previous allocation (labels) of alternatives that the algorithm will try to follow (classification problem). The default value is [].

* W = Optional argument. A list of weights for each criterion indicated by the decision maker. The default value is [], meaning that the algorithm will try to optimize this parameter.

* Q = Optional argument. The indifference threshold list indicated by the decision maker. The default value is [], meaning that the algorithm will try to optimize this parameter.

* P = Optional argument. The preference threshold list indicated by the decision maker. The default value is [], meaning that the algorithm will try to optimize this parameter.

* V = Optional argument. The veto threshold list indicated by the decision maker. The default value is [], meaning that the algorithm will try to optimize this parameter.

* cut_level = Optional argument. The list of possibles cut level values indicated by the decision maker. The default value is [0.5, 1.0], meaning that the algorithm will try to optimize this parameter with a value from 0.5 to 1.

* rule = Decides if the allocation rule is pessimist 'pc' or optimist 'oc'. The default values is 'pc'.

* number_of_classes = An integer that indicate the total number of classes of the problem. The default value is 2.

* elite = The quantity of best indivduals to be preserved in the genetic algorithm. The quantity should be low to avoid being traped in local otima. The default value is 1.

* mutation_rate = Chance to occur a mutation operation in the genetic algorithm. The default value is 0.01

* eta = Value of the mutation operator used in the genetic algorithm. The default value is 1.

* mu = Value of the breed operator used in the genetic algorithm. The default value is 2.

* population_size = The population size used in the genetic algorithm. The default value is 15.

* generations = The total number of iterations used in the genetic algorithm. The default value is 150.

* samples = The percentage of the number of alternatives (randomly selected) used in each submodel. The default value is 0.10.

* number_of_models = The total number of generated sub-models. The defaul value is 100.


<br/>__*"predict" arguments*__<br/>

* models = A list of optimized sub-models generated by the  "tree_electre_tri_b" function.

* dataset = A numpy array where the rows are the alternatives and columns are the criteria. 

* verbose = Prints the prediction for each alternative. The default value is True.

* rule = Decides if the allocation rule is pessimist 'pc' or optimist 'oc'. The default values is 'pc'.


<br/>__*"metrics" arguments. Returns the inferred parameters*__.<br/> 

* models = A list of optimized sub-models generated by the  "tree_electre_tri_b" function.

* number_of_classes = An integer that indicate the total number of classes of the problem. The default value is 2.


<br/>__*"plot_decision_boundaries" arguments*__<br/>

* data = A numpy array where the rows are the alternatives and columns are the criteria.  

* models = A list of optimized sub-models generated by the  "tree_electre_tri_b" function.
