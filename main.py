###############################################################################

# Required Libraries
import numpy as np

from electre_tree import tree_e_tri_b , util_e_tri_b
from sklearn.model_selection import train_test_split

###############################################################################


# EXAMPLE 1

###############################################################################

# Load Dataset
X = np.loadtxt('dataset-01.txt')

rule      = 'pc'
classes   = 4
target    = []
cut_level = [0.5, 1.0]
Q         = []
P         = []
V         = []
W         = []
B         = []

# Train Model
models = tree_e_tri_b.tree_electre_tri_b(X, target_assignment = target, W = W, Q = Q, P = P, V = V, B = B, cut_level = cut_level, rule = rule, number_of_classes = classes, elite = 1, eta = 1, mu = 2, population_size = 15, mutation_rate = 0.05, generations = 30, samples = 0.25, number_of_models = 1000)

# Predict
prediction, solutions = tree_e_tri_b.predict(models, X, verbose = False, rule = rule)

# Plot - Tree Model
util_e_tri_b.plot_points(X, prediction)

# Elicitated Paramneters
w_mean, w_std, q_mean, q_std, p_mean, p_std, v_mean, v_std, b_mean, b_std, cut_mean, cut_std, acc_mean, acc_std = tree_e_tri_b.metrics(models, number_of_classes = classes)

# Plot - Elicitated Parameters
e_tri = util_e_tri_b.electre_tri_b(X, W = w_mean, Q = q_mean, P = p_mean, V = v_mean, B = b_mean, cut_level = cut_mean, verbose = False, rule = rule, graph = True)

# Plot Tree Model - Decision Boundaries
tree_e_tri_b.plot_decision_boundaries(X, models)  

# Plot Mean Model - Decision Boundaries  
model_mean = []
model_mean.append([w_mean, acc_mean, [], [], [], b_mean, cut_mean, [], [], q_mean, p_mean, v_mean])
tree_e_tri_b.plot_decision_boundaries(X, model_mean)     

###############################################################################


# EXAMPLE 2

###############################################################################

# Load Dataset 2
dataset = np.loadtxt('dataset-02.txt')
X       = dataset [:,:-1]
y       = dataset [:,-1]

X_train, X_test, y_train,  y_test =  train_test_split(X, y, test_size = 0.50, random_state = 42)

rule      = 'pc'
classes   = 2
target    = list(y_train)
cut_level = [0.5, 1.0]
Q         = [0, 0, 0, 0]
P         = [0, 0, 0, 0]
V         = []
W         = []
B         = []

# Train Model
models = tree_e_tri_b.tree_electre_tri_b(X_train, target_assignment = target, W = W, Q = Q, P = P, V = V, B = B, cut_level = cut_level, rule = rule, number_of_classes = classes, elite = 1, eta = 1, mu = 2, population_size = 15, mutation_rate = 0.05, generations = 250, samples = 0.10, number_of_models = 1000)

# Predict
prediction, solutions = tree_e_tri_b.predict(models, X_test,  verbose = False, rule = rule)

# Plot - Tree Model
util_e_tri_b.plot_points(X_test, prediction)

# Elicitated Paramneters
w_mean, w_std, q_mean, q_std, p_mean, p_std, v_mean, v_std, b_mean, b_std, cut_mean, cut_std, acc_mean, acc_std = tree_e_tri_b.metrics(models, number_of_classes = classes)

# Plot - Elicitated Parameters
e_tri = util_e_tri_b.electre_tri_b(X, W = w_mean, Q = q_mean, P = p_mean, V = v_mean, B = b_mean, cut_level = cut_mean, verbose = False, rule = rule, graph = True)

# Plot Tree Model - Decision Boundaries
tree_e_tri_b.plot_decision_boundaries(X_test, models)  

# Plot Mean Model - Decision Boundaries  
model_mean = []
model_mean.append([w_mean, acc_mean, [], [], [], b_mean, cut_mean, [], [], q_mean, p_mean, v_mean])
tree_e_tri_b.plot_decision_boundaries(X_test, model_mean)     

###############################################################################


# EXAMPLE 3

###############################################################################

# Load Dataset 3
import pandas as pd
dataset   = pd.read_csv('dataset-03.txt', sep = '\t')
countries = dataset.iloc[:,0]
X         = dataset.iloc[:,1:].values

rule      = 'pc'
classes   = 4
target    = []
cut_level = [0.5, 1.0]
Q         = [0]*8
P         = [0]*8
V         = [2]*8
W         = []
B         = []

# Train Model
models = tree_e_tri_b.tree_electre_tri_b(X, target_assignment = target, W = W, Q = Q, P = P, V = V, B = B, cut_level = cut_level, rule = rule, number_of_classes = classes, elite = 1, eta = 1, mu = 2, population_size = 15, mutation_rate = 0.05, generations = 30, samples = 0.10, number_of_models = 500)

# Predict
prediction, solutions = tree_e_tri_b.predict(models, X,  verbose = False, rule = rule)

# Plot - Tree Model
util_e_tri_b.plot_points(X, prediction)

# Elicitated Paramneters
w_mean, w_std, q_mean, q_std, p_mean, p_std, v_mean, v_std, b_mean, b_std, cut_mean, cut_std, acc_mean, acc_std = tree_e_tri_b.metrics(models, number_of_classes = classes)

# Plot - Elicitated Parameters
e_tri = util_e_tri_b.electre_tri_b(X, W = w_mean, Q = q_mean, P = p_mean, V = v_mean, B = b_mean, cut_level = cut_mean, verbose = False, rule = rule, graph = True)

# Plot Tree Model - Decision Boundaries
tree_e_tri_b.plot_decision_boundaries(X, models)  

# Plot Mean Model - Decision Boundaries  
model_mean = []
model_mean.append([w_mean, acc_mean, [], [], [], b_mean, cut_mean, [], [], q_mean, p_mean, v_mean])
tree_e_tri_b.plot_decision_boundaries(X, model_mean)     

###############################################################################
