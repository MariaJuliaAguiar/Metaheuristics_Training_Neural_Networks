# import optimizers
from optimizers.pso import PSO
from optimizers.gwo import GWO
#import neural network - FeedForward
from nn.neural_network import NeuralNetwork
#import modules
import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Dataset
    dataset = "wine"
    
    ##Training and Testing

    xlsfile = pd.ExcelFile('datasets/'+ dataset + '/X_train.xlsx')
    X_train = xlsfile.parse('Sheet1')
    X_train = X_train[X_train.columns[1:]]
    X_train = X_train.to_numpy()
    X_train
    
    xlsfile1 = pd.ExcelFile('datasets/'+ dataset + '/X_test.xlsx')
    X_test = xlsfile1.parse('Sheet1')
    X_test = X_test[X_test.columns[1:]]
    X_test = X_test.to_numpy()
    X_test
    
    
    xlsfile2 = pd.ExcelFile('datasets/' + dataset +'/y_train.xlsx')
    y_train = xlsfile2.parse('Sheet1')
    y_train = y_train[y_train.columns[1:]]
    #y_train = y_train.to_numpy()
    y_train=y_train.values.flatten()
    
    
    xlsfile3 = pd.ExcelFile('datasets/' + dataset +'/y_test.xlsx')
    y_test = xlsfile3.parse('Sheet1')
    y_test = y_test[y_test.columns[1:]]
    #y_test = y_test.to_numpy()
    y_test = y_test.values.flatten()
    
    #initial position - Weigths and baias
    xlsfile = pd.ExcelFile('datasets/' + dataset +'/solucao_inicial.xlsx')
    sol = xlsfile.parse('Sheet1')
    sol = sol[sol.columns[1:]]
    initpos = sol.to_numpy()
    #network architecture
 
    n_inputs = 13 # features
    n_hidden = 20 
    n_classes = 3
    num_samples = X_train.shape[0] 
  
    #Simulation Parameters

    number_runs = 1
    number_iterations = 2
    dimensions = (n_inputs * n_hidden) + (n_hidden * n_hidden)+(n_hidden*n_classes) + 2*n_hidden + n_classes

    population_size = 100
    lb = [-2]
    up = [2]
    
    #PSO Hyperparameters

    c1 = 0.9
    c2 = 0.3
    Wmin = 0.9
    wMax = 0.9
    Vmax = 6
    
    ######## Run  ######
    hist_sim_pso = []
    best_sim_pso = []
    best_pso = []
     
    hist_sim_gwo = []
    best_gwo = []
    best_sim_gwo = []
    
    t1 = time.time()
    nn = NeuralNetwork(X_train,y_train,n_inputs,n_hidden,n_classes)
 
    for i in range(number_runs):
         print("Run : " + str(i))
         print("PSO")
         
         pso = PSO(population_size,dimensions,Vmax,wMax,Wmin,c1,c2,initpos)
         pso.optimizer(nn.forward_prop, number_iterations,lb,up)
         hist_sim_pso.append(pso.hist)
         best_sim_pso.append(pso.gbestpos)
         best_pso.append(pso.gbest)
          
          ##GWO
         print("GWO")
         gwo = GWO(dimensions,population_size,number_iterations,initpos)
         gwo.optimizer(nn.forward_prop, number_iterations,lb,up)
         hist_sim_gwo.append(gwo.hist)
         best_sim_gwo.append(gwo.alpha.position)
         best_gwo.append(gwo.alpha.obj_score)
    
    print("OI")
         

    # Save Results
    results_path = "./results/"
    print("\n")
    #Accuracy
    min_ind_PSO = np.argmin(best_pso)
    acc_PSO = (nn.predict(best_sim_pso[min_ind_PSO],X_test) ==y_test).mean()
    print("PSO Accuracy: " + str(acc_PSO))
    print("PSO Best fitness:" + str(best_pso[min_ind_PSO]))

    
    min_ind_Gwo = np.argmin(best_gwo)
    acc_GWO = (nn.predict(best_sim_gwo[min_ind_Gwo],X_test) ==y_test).mean()
    print(" \nGWO Accuracy: " + str(acc_GWO))
    print("Gwo Best fitness:" + str(best_gwo[min_ind_PSO]))

   
    # Convergence curves - Plots
    numpy_array_PSO = np.array(hist_sim_pso)
    transpose_PSO = numpy_array_PSO.T
    numpy_array_GWO = np.array(hist_sim_gwo)
    transpose_GWO = numpy_array_GWO.T
    # Plot current gBest
    matplotlib.use('Agg')
    plt.plot(transpose_PSO)
    plt.xlabel("Iteration")
    plt.ylabel("gBest acc")
    plt.savefig(results_path + "PSO_convergence"  + ".png")
    plt.close()
    
    matplotlib.use('Agg')
    plt.plot(transpose_GWO)
    plt.xlabel("Iteration")
    plt.ylabel("gBest acc")
    plt.savefig(results_path + "GWO_convergence"  + ".png")
    plt.close()
    
    # Best model - weigths and baias
     # Save Weigths and baias
    best_PSO  = pd.DataFrame(best_sim_pso[min_ind_PSO],columns =['BestPSO'])
    best_PSO.to_excel(results_path +"Best_PSO.xlsx")
    best_GWO  = pd.DataFrame(best_sim_gwo[min_ind_Gwo],columns =['BestGWO'])
    best_GWO.to_excel(results_path + "best_GWO.xlsx")
    tempoExec = time.time() - t1
    print("Tempo de execução: {} segundos".format(tempoExec))
    print("Fimm")
