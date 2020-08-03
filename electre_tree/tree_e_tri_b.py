###############################################################################

import copy
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from electre_tree.util_e_tri_b import genetic_algorithm, electre_tri_b
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

###############################################################################

# Main    
def tree_electre_tri_b(dataset, target_assignment = [], W = [], Q = [], P = [], V = [], B = [], cut_level = [0.5, 1.0], rule = 'pc', number_of_classes = 2, elite = 1, eta = 1, mu = 2, population_size = 25, mutation_rate = 0.01, generations = 150, samples = 0.10, number_of_models = 100):
    count           = 0
    ensemble_model  = []
    if (len(target_assignment) == 0):
        kmeans   = KMeans(n_clusters = number_of_classes, init = 'k-means++', n_init = 10, max_iter = 200, random_state = 42)
        y        = kmeans.fit(dataset)
        idx      = np.argsort(y.cluster_centers_.sum(axis = 1))[::-1]
        idc      = np.zeros_like(idx)
        idc[idx] = np.arange(number_of_classes)
        y        = idc[y.labels_]
    elif (len(target_assignment) > 0):
        y        = target_assignment
    def target_function(variable_list):
        variable_list = variable_list.tolist()
        W_t = variable_list[0:random_dataset.shape[1]]
        Q_t = variable_list[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        P_t = variable_list[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        V_t = variable_list[random_dataset.shape[1]*3:random_dataset.shape[1]*4]
        B_t = []
        if (number_of_classes - 1 == 1):
            B_t  = variable_list[random_dataset.shape[1]*4:random_dataset.shape[1]*5]
        else:
            for i in range(0, (number_of_classes - 1)):
                B_t.append(variable_list[random_dataset.shape[1]*(i+4):random_dataset.shape[1]*(i+5)])
        ctv   = variable_list[-1]
        e_tri = electre_tri_b(random_dataset, Q = Q_t, P = P_t, V = V_t, W = W_t, B = B_t, cut_level = ctv, verbose = False, rule = rule)
        cost = 0
        for i in range(0, len(e_tri)):
            if (abs(e_tri[i] - random_y[i]) != 0):
                cost = cost + 1
        return cost #1 - accuracy(e_tri, random_y, clstr) 
    while count < number_of_models:
        random_dataset = np.copy(dataset)
        random_y       = np.copy(y)
        random_W       = copy.deepcopy(W)
        random_Q       = copy.deepcopy(Q)
        random_P       = copy.deepcopy(P)
        random_V       = copy.deepcopy(V)
        random_B       = copy.deepcopy(B)
        if (random_dataset.shape[1] > 2):
            criteria_remove = random.sample(list(range(0, dataset.shape[1])), random.randint(1, dataset.shape[1]- 2))
            random_dataset  = np.delete(random_dataset, criteria_remove, axis = 1)
        else:
            criteria_remove = []
        criteria_remove.sort(reverse = True)
        for i in range(dataset.shape[1] - 1, -1, -1):
            if i in criteria_remove:
                if (len(random_W) > 0):
                    del random_W[i]
                if (len(random_Q) > 0):
                    del random_Q[i]
                if (len(random_P) > 0):
                    del random_P[i]
                if (len(random_V) > 0):
                    del random_V[i]
                if (len(random_B) > 0 and number_of_classes == 2):
                    del random_B[0][i]
                elif(len(random_B) > 0 and number_of_classes > 2):
                    for j in range(0, (number_of_classes - 1)):
                        del random_B[j][i]
        criteria     =  [item for item in list(range(0, dataset.shape[1])) if item not in criteria_remove]
        cases_remove = random.sample(list(range(0, dataset.shape[0])), math.ceil(dataset.shape[0]*(1 - samples))) 
        while (dataset.shape[0] - len(cases_remove) < number_of_classes):
            del cases_remove[-1]            
        if (len(cases_remove) > 0):
            random_dataset  = np.delete(random_dataset, cases_remove, axis = 0)
            random_y        = np.delete(random_y, cases_remove, axis = 0)
            random_y        = list(random_y)
        if (len(random_W) == 0):  
            min_values = [0.00]*random_dataset.shape[1]
            max_values = [1.00]*random_dataset.shape[1]
        elif (len(random_W) > 0):
            min_values = copy.deepcopy(random_W)
            max_values = copy.deepcopy(random_W)
        if (len(random_Q) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_Q) > 0):
            min_values.extend(copy.deepcopy(random_Q))
            max_values.extend(copy.deepcopy(random_Q))
        if (len(random_P) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_P) > 0):
            min_values.extend(copy.deepcopy(random_P))
            max_values.extend(copy.deepcopy(random_P))
        if (len(random_V) == 0):  
            min_values.extend([0.00]*random_dataset.shape[1])
            max_values.extend(list(np.amax(random_dataset, axis = 0) - np.amin(random_dataset, axis = 0)))
        elif (len(random_V) > 0):
            min_values.extend(copy.deepcopy(random_V))
            max_values.extend(copy.deepcopy(random_V))
        if (len(random_B) == 0):
            for i in range(0, (number_of_classes - 1)):
                min_values.extend(list(np.amin(random_dataset, axis = 0)))
                max_values.extend(list(np.amax(random_dataset, axis = 0)))
        elif (len(random_B) > 0):
            for i in range(0, (number_of_classes - 1)):
                min_values.extend(copy.deepcopy(random_B[i]))
                max_values.extend(copy.deepcopy(random_B[i]))
        min_values.extend([cut_level[0]])
        max_values.extend([cut_level[1]])
        ga = genetic_algorithm(population_size = population_size, mutation_rate = mutation_rate, elite = elite, min_values = min_values, max_values = max_values, eta = eta, mu = mu, generations = generations, size = random_dataset.shape[1], target_function = target_function)
        W_ga = ga[0:random_dataset.shape[1]]
        Q_ga = ga[random_dataset.shape[1]*1:random_dataset.shape[1]*2]
        P_ga = ga[random_dataset.shape[1]*2:random_dataset.shape[1]*3]
        V_ga = ga[random_dataset.shape[1]*3:random_dataset.shape[1]*4]
        B_ga = []
        if (number_of_classes - 1 == 1):
            B_ga = ga[random_dataset.shape[1]*4:random_dataset.shape[1]*5].tolist()
        else:
            for i in range(0, (number_of_classes - 1)):
                B_ga.append(ga[random_dataset.shape[1]*(i+4):random_dataset.shape[1]*(i+5)].tolist())
        acc   = round( (1 - (ga[-1]/random_dataset.shape[0])),2)
        ctv   = ga[-2]
        y_hat = electre_tri_b(random_dataset, W = W_ga, Q = Q_ga, P = P_ga, V = V_ga, B = B_ga, cut_level = ctv, verbose = False, rule = rule)
        ensemble_model.append([W_ga, acc, criteria, criteria_remove, cases_remove, B_ga, ctv, y_hat, random_y, Q_ga, P_ga, V_ga])
        count = count + 1
        print('Model # ' + str(count)) 
    return ensemble_model

###############################################################################
    
# Prediction
def predict(models, dataset, verbose = True, rule = 'pc'):
    prediction     = []
    solutions      = [[]]*dataset.shape[0]
    ensemble_model = copy.deepcopy(models)
    for j in range(0, dataset.shape[0]):
        for i in range(0, len(ensemble_model)):
            alternative = dataset[j,:].reshape((1, dataset.shape[1]))   
            alternative = np.delete(alternative, ensemble_model[i][3], axis = 1)
            e_tri       = electre_tri_b(alternative, Q = ensemble_model[i][9], P = ensemble_model[i][10], V = ensemble_model[i][11], W = ensemble_model[i][0], B = ensemble_model[i][5], cut_level = ensemble_model[i][6], verbose = False, rule = rule)
            if (i == 0):
                solutions[j] = e_tri
            else:
                solutions[j].extend(e_tri)
        lst_count = [x for x in set(solutions[j]) if solutions[j].count(x) > 1]
        max_k       =  0
        max_k_value = -1
        if (len(lst_count) == 0):
            lst_count = [1]*dataset.shape[0]
            lst_count = [item for sublist in solutions for item in sublist]
        for k in range (0, len(lst_count)):
            if (solutions[j].count(lst_count[k]) > max_k_value):
                max_k_value = solutions[j].count(lst_count[k]) 
                max_k       = k  
        prediction.append(lst_count[max_k])
        if (verbose == True):
            print('a' + str(j + 1) + ' = ' + str(lst_count[max_k]))
    return prediction, solutions

###############################################################################

# Metrics
def metrics(models, number_of_classes):
    ensemble_model      = copy.deepcopy(models)   
    number_profiles     = number_of_classes - 1
    features_importance = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    count_features      = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    mean_features       = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    std_features        = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    profiles_importance = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))*number_profiles
    count_profiles      = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))*number_profiles
    mean_profiles       = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))*number_profiles
    std_profiles        = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))*number_profiles
    q_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    q_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    q_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    p_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    v_tresholds         = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    v_tresholds_mean    = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    v_tresholds_std     = [0]*(len(ensemble_model[0][2]) + len(ensemble_model[0][3]))
    acc_mean            = 0
    acc_std             = 0
    cut_mean            = 0
    cut_std             = 0
    for i in range(0, len(ensemble_model)):
        weights  = ensemble_model[i][0]   
        criteria = ensemble_model[i][2] 
        q        = ensemble_model[i][9] 
        p        = ensemble_model[i][10] 
        v        = ensemble_model[i][11] 
        if (number_profiles > 1):
            profiles = ensemble_model[i][5]
        else:
            profiles = [[]]*1
            profiles[0] = ensemble_model[i][5]
        acc_mean = acc_mean + ensemble_model[i][1]
        cut_mean = cut_mean + ensemble_model[i][6]
        for j in range(0, len(criteria)):
            features_importance[criteria[j]] = features_importance[criteria[j]] + weights[j] 
            q_tresholds[criteria[j]]         = q_tresholds[criteria[j]] + q[j]
            p_tresholds[criteria[j]]         = p_tresholds[criteria[j]] + p[j]
            v_tresholds[criteria[j]]         = v_tresholds[criteria[j]] + v[j]
            count_features[criteria[j]]      = count_features[criteria[j]] + 1
        for j in range(0, number_profiles):
            for k in range(0, len(criteria)):
                profiles_importance[criteria[k] + len(features_importance)*j] = profiles_importance[criteria[k] + len(features_importance)*j]  + profiles[j][k]
                count_profiles [criteria[k] + len(features_importance)*j]     = count_profiles [criteria[k] + len(features_importance)*j]  + 1
    acc_mean = acc_mean/len(ensemble_model)
    cut_mean = cut_mean/len(ensemble_model)
    for i in range(0, len(mean_features)):
        mean_features[i]    = features_importance[i]/count_features[i]
        q_tresholds_mean[i] = q_tresholds[i]/count_features[i]
        p_tresholds_mean[i] = p_tresholds[i]/count_features[i]
        v_tresholds_mean[i] = v_tresholds[i]/count_features[i]
    for i in range(0, len(mean_profiles)):
        mean_profiles[i] = profiles_importance[i]/count_profiles[i]
    for i in range(0, len(ensemble_model)):  
        weights  = ensemble_model[i][0] 
        criteria = ensemble_model[i][2] 
        q        = ensemble_model[i][9] 
        p        = ensemble_model[i][10] 
        v        = ensemble_model[i][11] 
        if (number_profiles > 1):
            profiles = ensemble_model[i][5]
        else:
            profiles    = [[]]*1
            profiles[0] = ensemble_model[i][5]
        acc_std  = acc_std + (ensemble_model[i][1] - acc_mean)**2
        cut_std  = cut_std + (ensemble_model[i][6] - cut_mean)**2
        for j in range(0, len(criteria)):
            std_features[criteria[j]]    = std_features[criteria[j]]    + (weights[j] - mean_features[criteria[j]])**2
            q_tresholds_std[criteria[j]] = q_tresholds_std[criteria[j]] + (q[j] - q_tresholds_mean[criteria[j]])**2
            p_tresholds_std[criteria[j]] = p_tresholds_std[criteria[j]] + (p[j] - p_tresholds_mean[criteria[j]])**2
            v_tresholds_std[criteria[j]] = v_tresholds_std[criteria[j]] + (v[j] - v_tresholds_mean[criteria[j]])**2
        for j in range(0, number_profiles):
            for k in range(0, len(criteria)):
                std_profiles[criteria[k] + len(features_importance)*j] = std_profiles[criteria[k] + len(features_importance)*j]  + (profiles[j][k] -  mean_profiles[criteria[k] + len(features_importance)*j])**2
    acc_std  = (acc_std/(len(ensemble_model)-1))**(1/2)
    cut_std  = (cut_std/(len(ensemble_model)-1))**(1/2)
    for i in range(0, len(std_features)): 
         std_features[i]    = (std_features[i]/(count_features[i]-1))**(1/2)
         q_tresholds_std[i] = (q_tresholds_std[i]/(count_features[i]-1))**(1/2)
         p_tresholds_std[i] = (p_tresholds_std[i]/(count_features[i]-1))**(1/2)
         v_tresholds_std[i] = (v_tresholds_std[i]/(count_features[i]-1))**(1/2)
    for i in range(0, len(std_profiles)): 
         std_profiles[i] = (std_profiles[i]/(count_profiles[i]-1))**(1/2)
    profile_mean_list = []
    profile_std_list  = []
    for i in range(0, number_profiles):
        profile_mean_list.append(mean_profiles[(i + (len(features_importance)-1)*i):(len(features_importance) + (len(features_importance))*i)])
        profile_std_list.append(std_profiles[(i + (len(features_importance)-1)*i):(len(features_importance) + (len(features_importance))*i)])
    return mean_features, std_features, q_tresholds_mean, q_tresholds_std, p_tresholds_mean, p_tresholds_std, v_tresholds_mean, v_tresholds_std, profile_mean_list, profile_std_list, cut_mean, cut_std, acc_mean, acc_std

###############################################################################
 
# Function: Plot Decision Boundaries
def plot_decision_boundaries(data, models):
    plt.style.use('ggplot')
    colors = {'A':'#bf77f6', 'B':'#ff9408', 'C':'#d1ffbd', 'D':'#c85a53', 'E':'#3a18b1', 'F':'#ff796c', 'G':'#04d8b2', 'H':'#ffb07c', 'I':'#aaa662', 'J':'#0485d1', 'K':'#fffe7a', 'L':'#b0dd16', 'M':'#85679', 'N':'#12e193', 'O':'#82cafc', 'P':'#ac9362', 'Q':'#f8481c', 'R':'#c292a1', 'S':'#c0fa8b', 'T':'#ca7b80', 'U':'#f4d054', 'V':'#fbdd7e', 'W':'#ffff7e', 'X':'#cd7584', 'Y':'#f9bc08', 'Z':'#c7c10c'}
    #variance  = 0
    if (data.shape[1] == 2):
        xpts      = np.arange(data[:,0].min(), data[:,0].max(), (data[:,0].max()-data[:,0].min())/100)
        ypts      = np.arange(data[:,1].min(), data[:,1].max(), (data[:,1].max()-data[:,1].min())/100)
        points    = np.array(list(itertools.product(xpts, ypts)))
        points_in = points
    else:
        tSVD      = TruncatedSVD(n_components = 2, n_iter = 100, random_state = 42)
        min_value = np.min(data, axis = 0)
        max_value = np.max(data, axis = 0)
        tSVD_proj = tSVD.fit_transform(np.vstack((data, min_value, max_value)))
        xpts      = np.arange(tSVD_proj[:,0].min(), tSVD_proj[:,0].max(), (data[:,0].max()-data[:,0].min())/100)
        ypts      = np.arange(tSVD_proj[:,1].min(), tSVD_proj[:,1].max(), (data[:,1].max()-data[:,1].min())/100)
        points    = np.array(list(itertools.product(xpts, ypts)))
        points_in = tSVD.inverse_transform(points)
        #variance  = sum(np.var(tSVD_proj, axis = 0) / np.var(tSVD_proj, axis = 0).sum())
    prediction, _ = predict(models, points_in, verbose = False)
    class_list    = list(set(prediction))
    for i in range(0, len(prediction)): 
        prediction[i] = str(prediction[i])      
    for i in range(0, len(prediction)):
        for j in range(0, len(class_list)):
            prediction[i] = prediction[i].replace(str(class_list[j]), str(chr(ord('A') + class_list[j])))      
    class_list = list(set(prediction))
    class_list.sort()  
    fig, ax = plt.subplots()
    ax.scatter(points[:,0], points[:,1], c = [colors[k] for k in prediction], alpha = 0.5, s = 120, marker = 's')
    #if (variance > 0):
        #plt.xlabel('EV: ' + str(round(variance*100, 2)) + '%')
    plt.show()  
    return

###############################################################################
