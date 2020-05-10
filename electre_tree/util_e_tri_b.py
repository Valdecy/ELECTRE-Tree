###############################################################################

# Required Libraries
import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy  as np
import random
import os

from sklearn.decomposition import TruncatedSVD

###############################################################################

# Function: Target
def target_function():
    return

# Function: Clip Thresholds
def clip_thresholds(population, indv = 0, size = 3, min_values = [-5,-5], max_values = [5,5]):
    size = size
    for i in range(0, size):
        if (population[indv][size*1 + i] > population[indv][size*2 + i]):
            population[indv][size*1 + i] = population[indv][size*2 + i]
        if (population[indv][size*2 + i] > population[indv][size*3 + i] and population[indv][size*1 + i] > population[indv][size*3 + i]):
            population[indv][size*1 + i] = population[indv][size*3 + i]
            population[indv][size*2 + i] = population[indv][size*3 + i]      
        if (population[indv][size*2 + i] > population[indv][size*3 + i] and population[indv][size*1 + i] < population[indv][size*3 + i]):
            population[indv][size*2 + i] = population[indv][size*3 + i] 
    population[indv,:-1] = np.clip(population[indv,:-1], min_values, max_values)
    return population

# Function: Clip Profiles
def clip_profiles(population, indv = 0, size = 3, min_values = [-5,-5], max_values = [5,5]):
    profile_rows = int(population.shape[1] - 2)
    profile_cols = size
    for i in range(int((profile_cols*5)-1), profile_rows):
        if (population[indv][i] < population[indv][i - profile_cols]):
            population[indv][i] = population[indv][i - profile_cols]
    population[indv,:-1] = np.clip(population[indv,:-1], min_values, max_values)
    return population

# Function: Initialize Variables
def initial_population(population_size = 5, size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population = clip_thresholds(population, indv = i, size = size, min_values = min_values, max_values = max_values)
        population = clip_profiles(population  , indv = i, size = size, min_values = min_values, max_values = max_values)
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], size = 3, mu = 1, elite = 0, target_function = target_function):
    offspring = np.copy(population)
    b_offspring = 0
    if (elite > 0):
        preserve = np.copy(population[population[:,-1].argsort()])
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring[i,j] = preserve[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        offspring = clip_thresholds(offspring, indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring = clip_profiles(offspring  , indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1]) 
    return offspring
 
# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])
        offspring = clip_thresholds(offspring, indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring = clip_profiles(offspring  , indv = i, size = size, min_values = min_values, max_values = max_values)
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])                        
    return offspring

# Function: GA
def genetic_algorithm(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, mu = 1, generations = 50, size = 3, target_function = target_function):    
    count = 0
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, size = size, target_function = target_function)
    fitness = fitness_function(population)    
    elite_ind = np.copy(population[population[:,-1].argsort()][0,:])
    while (count <= generations):  
        offspring = breeding(population, fitness, min_values = min_values, max_values = max_values, mu = mu, elite = elite, size = size, target_function = target_function) 
        population = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, size = size, target_function = target_function)
        fitness = fitness_function(population)
        value = np.copy(population[population[:,-1].argsort()][0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        count = count + 1      
    return elite_ind 

###############################################################################
    
# Function: Concordance Matrices and Vectors
def concordance_matrices_vectors(performance_matrix, number_of_profiles, number_of_alternatives, B, P, Q, W):         
    n_rows = number_of_profiles * number_of_alternatives
    n_cols = performance_matrix.shape[1]
    # Concordance Matrix x_b
    concordance_matrix = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, concordance_matrix .shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, concordance_matrix.shape[1]):
            if (B[count, j] - performance_matrix[alternative, j] >= P[0, j]):
                concordance_matrix[i, j] = 0
            elif (B[count, j] - performance_matrix[alternative, j] < Q[0, j]):
                concordance_matrix[i, j] = 1
            else:
                concordance_matrix[i, j] = (P[0, j] - B[count, j] + performance_matrix[alternative, j])/(P[0, j] - Q[0, j])     
    # Concordance Vector x_b
    concordance_vector = np.zeros((n_rows, 1))
    for i in range(0, concordance_vector.shape[0]):
        for j in range(0, concordance_matrix.shape[1]):
            concordance_vector[i, 0] = concordance_vector[i, 0] + concordance_matrix[i, j]*W[j]
        if (W.sum(axis = 0) != 0):
            concordance_vector[i, 0] = concordance_vector[i, 0]/W.sum(axis = 0)           
    # Concordance Matrix b_x
    concordance_matrix_inv = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, concordance_matrix_inv.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, concordance_matrix_inv.shape[1]):
            if (-B[count, j] + performance_matrix[alternative, j] >= P[0, j]):
                concordance_matrix_inv[i, j] = 0
            elif (-B[count, j] + performance_matrix[alternative, j] < Q[0, j]):
                concordance_matrix_inv[i, j] = 1
            else:
                concordance_matrix_inv[i, j] = (P[0, j] + B[count, j] - performance_matrix[alternative, j])/(P[0, j] - Q[0, j])        
    # Concordance Vector b_x
    concordance_vector_inv = np.zeros((n_rows, 1))
    for i in range(0, concordance_vector_inv.shape[0]):
        for j in range(0, concordance_matrix_inv.shape[1]):
            concordance_vector_inv[i, 0] = concordance_vector_inv[i, 0] + concordance_matrix_inv[i, j]*W[j]
        if (W.sum(axis = 0) != 0):
            concordance_vector_inv[i, 0] = concordance_vector_inv[i, 0]/W.sum(axis = 0)    
    return concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv

# Function: Discordance Matrices
def discordance_matrices(performance_matrix, number_of_profiles, number_of_alternatives, B, P, V):
    n_rows = number_of_profiles * number_of_alternatives
    n_cols = performance_matrix.shape[1]
    # Discordance Matrix x_b
    disconcordance_matrix = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, disconcordance_matrix.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, disconcordance_matrix.shape[1]):
            if (B[count, j] - performance_matrix[alternative, j] < P[0, j]):
                disconcordance_matrix[i, j] = 0
            elif (B[count, j] - performance_matrix[alternative, j] >= V[0, j]):
                disconcordance_matrix[i, j] = 1
            else:
                disconcordance_matrix[i, j] = (-P[0, j] + B[count, j] - performance_matrix[alternative, j])/(V[0, j] - P[0, j])  
    # Discordance Matrix b_x
    disconcordance_matrix_inv = np.zeros((n_rows, n_cols))    
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, disconcordance_matrix_inv.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, disconcordance_matrix_inv.shape[1]):
            if (-B[count, j] + performance_matrix[alternative, j] < P[0, j]):
                disconcordance_matrix_inv[i, j] = 0
            elif (-B[count, j] + performance_matrix[alternative, j] >= V[0, j]):
                disconcordance_matrix_inv[i, j] = 1
            else:
                disconcordance_matrix_inv[i, j] = (-P[0, j] - B[count, j] + performance_matrix[alternative, j])/(V[0, j] - P[0, j])
    return disconcordance_matrix, disconcordance_matrix_inv

# Function: Credibility Vectors
def credibility_vectors(number_of_profiles, number_of_alternatives, concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv, disconcordance_matrix, disconcordance_matrix_inv):
    n_rows = number_of_profiles * number_of_alternatives  
    # Credibility Vector x_b
    credibility_vector = np.zeros((n_rows, 1))
    for i in range(0, credibility_vector.shape[0]):
        credibility_vector[i, 0] = concordance_vector[i, 0]
        for j in range(0, concordance_matrix.shape[1]):
            if (disconcordance_matrix[i, j] > concordance_vector[i, 0]):
                value = (1 - disconcordance_matrix[i, j])/(1 - concordance_vector[i, 0])
                credibility_vector[i, 0] = credibility_vector[i, 0]*value  
    # Credibility Vector b_x        
    credibility_vector_inv = np.zeros((n_rows, 1))
    for i in range(0, credibility_vector_inv.shape[0]):
        credibility_vector_inv[i, 0] = concordance_vector_inv[i, 0]
        for j in range(0, concordance_matrix_inv.shape[1]):
            if (disconcordance_matrix_inv[i, j] > concordance_vector_inv[i, 0]):
                value = (1 - disconcordance_matrix_inv[i, j])/(1 - concordance_vector_inv[i, 0])
                credibility_vector_inv[i, 0] = credibility_vector_inv[i, 0]*value
    return credibility_vector, credibility_vector_inv

# Function: Fuzzy Logic
def fuzzy_logic(number_of_profiles, number_of_alternatives, credibility_vector, credibility_vector_inv, cut_level):
    n_rows = number_of_profiles * number_of_alternatives 
    fuzzy_vector = []
    fuzzy_matrix = [[]]* number_of_alternatives 
    for i in range(0, n_rows):
        if (credibility_vector[i, 0] >= cut_level and credibility_vector_inv[i, 0] >= cut_level):
            fuzzy_vector.append('I')
        elif (credibility_vector[i, 0] >= cut_level and credibility_vector_inv[i, 0] <  cut_level):
            fuzzy_vector.append('>')
        elif (credibility_vector[i, 0] <  cut_level and credibility_vector_inv[i, 0] >= cut_level):
            fuzzy_vector.append('<')
        elif (credibility_vector[i, 0] <  cut_level and credibility_vector_inv[i, 0] <  cut_level):
            fuzzy_vector.append('R')
    
    fm = [fuzzy_vector[x:x+number_of_alternatives] for x in range(0, len(fuzzy_vector), number_of_alternatives)]
    for j in range(number_of_profiles-1, -1,-1):
        for i in range(0, number_of_alternatives):
            fuzzy_matrix[i] = fuzzy_matrix[i] + [fm[j][i]]
    return fuzzy_matrix

# Function: Classification
def classification_algorithm(number_of_profiles, number_of_alternatives, fuzzy_matrix, rule, verbose = True):
    classification = []
    if (rule == 'pc'):
        # Pessimist Classification
        for i1 in range(0, number_of_alternatives):
            class_i = number_of_profiles
            count   = 0
            for i2 in range(0, number_of_profiles):
                count = count + 1
                if (fuzzy_matrix[i1][i2] == '>'):
                    class_i = int(number_of_profiles - count)
            classification.append(class_i)
            if (verbose == True):
                print('a' + str(i1 + 1) + ' = ' + 'C' + str(class_i))  
    elif(rule == 'oc'):
        # Optimistic Classification
        for i1 in range(0, number_of_alternatives):
            class_i = 0
            count   = 0
            for i2 in range(number_of_profiles - 1, -1, -1):
                count = count + 1
                if (fuzzy_matrix[i1][i2] == '<'):
                    class_i = int(count)
            classification.append(class_i)
            if (verbose == True):
                print('a' + str(i1 + 1) + ' = ' + 'C' + str(class_i))    
    return classification   

# Function: Plot Projected Points 
def plot_points(data, classification):
    plt.style.use('ggplot')
    #'D':'#c85a53'
    #'B':'#ff9408'
    colors = {'A':'#bf77f6', 'B':'#fed8b1', 'C':'#d1ffbd', 'D':'#f08080', 'E':'#3a18b1', 'F':'#ff796c', 'G':'#04d8b2', 'H':'#ffb07c', 'I':'#aaa662', 'J':'#0485d1', 'K':'#fffe7a', 'L':'#b0dd16', 'M':'#85679', 'N':'#12e193', 'O':'#82cafc', 'P':'#ac9362', 'Q':'#f8481c', 'R':'#c292a1', 'S':'#c0fa8b', 'T':'#ca7b80', 'U':'#f4d054', 'V':'#fbdd7e', 'W':'#ffff7e', 'X':'#cd7584', 'Y':'#f9bc08', 'Z':'#c7c10c'}
    classification_ = copy.deepcopy(classification)
    color_leg = {}
    #variance  = 0
    if (data.shape[1] == 2):
        data_proj = np.copy(data)
    else:
        tSVD      = TruncatedSVD(n_components = 2, n_iter = 100, random_state = 42)
        tSVD_proj = tSVD.fit_transform(data)
        data_proj = np.copy(tSVD_proj)
        #variance  = sum(np.var(tSVD_proj, axis = 0) / np.var(tSVD_proj, axis = 0).sum())
    class_list  = list(set(classification_))
    for i in range(0, len(classification_)):
        classification_[i] = str(classification_[i])
    for i in range(0, len(classification_)):
        for j in range(0, len(class_list)):
            classification_[i] = classification_[i].replace(str(class_list[j]), str(chr(ord('A') + class_list[j])))
    class_list = list(set(classification_))
    class_list.sort() 
    for i in range(0, len(class_list)):
        color_leg[class_list[i]] = colors[class_list[i]]
    patchList = []
    for key in color_leg:
        data_key = mpatches.Patch(color = color_leg[key], label = key)
        patchList.append(data_key)
    for i in range(0, data_proj.shape[0]):
        plt.text(data_proj[i, 0], data_proj[i, 1], 'x' + str(i+1), size = 10, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = colors[classification_[i]],))
    plt.gca().legend(handles = patchList, loc = 'center left', bbox_to_anchor = (1.05, 0.5))
    axes = plt.gca()
    xmin = np.amin(data_proj[:,0])
    xmax = np.amax(data_proj[:,0])
    axes.set_xlim([xmin*0.7, xmax*1])
    ymin = np.amin(data_proj[:,1])
    ymax = np.amax(data_proj[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin*0.7, ymax*1])
    #if (variance > 0):
        #plt.xlabel('EV: ' + str(round(variance*100, 2)) + '%')
    plt.show()
    return
    
# Function: ELECTRE TRI-B
def electre_tri_b(performance_matrix, W = [], Q = [], P = [], V = [], B = [], cut_level = 1.0, verbose = True, rule = 'pc', graph = False):  
    # Loading Parameters
    if (isinstance(B[0], list)):
        number_of_profiles = len(B)
    else:
        number_of_profiles = 1
    number_of_alternatives = performance_matrix.shape[0]    
    p_vector = np.zeros((1, performance_matrix.shape[1]))
    q_vector = np.zeros((1, performance_matrix.shape[1]))
    v_vector = np.zeros((1, performance_matrix.shape[1]))    
    for i in range(0, p_vector.shape[1]):
        p_vector[0][i] = P[i]
        q_vector[0][i] = Q[i]
        v_vector[0][i] = V[i]   
    w_vector = np.array(W)
    b_matrix = np.array(B)
    if (isinstance(B[0], list)):
        b_matrix = np.array(B) 
    else:
        b_matrix = np.zeros((1, performance_matrix.shape[1]))
        for i in range(0, performance_matrix.shape[1]):
            b_matrix[0][i] = B[i]
     
    # Algorithm       
    concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv = concordance_matrices_vectors(performance_matrix = performance_matrix, number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, B = b_matrix, P = p_vector, Q = q_vector, W = w_vector)    
    
    disconcordance_matrix, disconcordance_matrix_inv = discordance_matrices(performance_matrix = performance_matrix, number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, B = b_matrix, P = p_vector, V = v_vector)    
    
    credibility_vector, credibility_vector_inv = credibility_vectors(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, concordance_matrix = concordance_matrix, concordance_matrix_inv = concordance_matrix_inv, concordance_vector = concordance_vector, concordance_vector_inv = concordance_vector_inv, disconcordance_matrix = disconcordance_matrix, disconcordance_matrix_inv = disconcordance_matrix_inv) 
    
    fuzzy_matrix = fuzzy_logic(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, credibility_vector = credibility_vector, credibility_vector_inv = credibility_vector_inv, cut_level = cut_level)  
    
    classification = classification_algorithm(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, fuzzy_matrix = fuzzy_matrix, rule = rule, verbose = verbose)
    
    if (graph == True):
        plot_points(performance_matrix, classification)
    return classification

###############################################################################

# Function: Trim Models
def trim_models(models, target = 0.95):
    for i in range(len(models) -1, -1, -1):
        if (models[i][1] < target):
            del models[i]
    return models

############################################################################### 
   
# Function: Align Clusters  (Kronecker Delta Distance). Simdist = (label_1 x label_2)
def align_clusters(label_1, label_2):
    k          = int(np.max(label_2) + 1)
    label_2_re = np.copy(label_2)
    simdist    = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            eci = [num for num in label_1 if num == i] # Elements Cluster i
            ecj = [num for num in label_2 if num == j] # Elements Cluster j
            ici = [idx for idx in range(len(label_1)) if label_1[idx] == i] # Index Cluster i
            icj = [idx for idx in range(len(label_2)) if label_2[idx] == j] # Index Cluster j
            cme = len(set(ici) & set(icj)) # Common
            if (len(eci) == 0):
                eci = [1]
            if (len(ecj) == 0):
                ecj = [1]
            simdist[i, j] = (cme/len(eci) + cme/len(ecj))/2
    for m in range(0, k):   
        idx        = np.argmax(simdist[m,:])
        label_2_re = [str(m) + 'r' if i == idx else i for i in label_2_re]
        simdist[:, idx] = -1
    for n in range(0, len(label_2_re)):
        if (isinstance(label_2_re[n], str)):
            label_2_re[n] = label_2_re[n].replace('r', '')
    label_2_re = list(map(int, label_2_re))
    return label_2_re

###############################################################################

# Function: Accuracy
def accuracy(y_hat, random_y, align = False):
    if(align == True):
        random_y = align_clusters(y_hat, random_y)
    acc   = 0
    for i in range(0, len(random_y)):
        diff = y_hat[i] - random_y[i]
        if (diff == 0):
            acc = acc + 1
    return round(acc/len(random_y), 2)

###############################################################################