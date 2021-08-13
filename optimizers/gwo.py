import pandas as pd
import numpy as np
import random

class wolf:
    
    def __init__(self, dim):
        self.position =  np.zeros((dim,))
        self.obj_score =float("inf")
class GWO:
    def __init__(self, dim, wolves_n, iter_n,initpos):
        self.wolves = np.array(
            [wolf(dim) for i in range(wolves_n)]
        )
        self.alpha = wolf(dim)
        self.beta = wolf(dim)
        self.delta = wolf(dim)
        self.hist = []
        self.dim = dim
        self.initpos = initpos

    def gerar_populacao(self):
        
        i=0
        for w in self.wolves:
            w.position = self.initpos[i]
            i=i+1
        #self.alpha,self.beta,self.delta = self.wolves[:3]
        return
    
    # obj_fun   - objective function to be minimized
    # soln_dim  - dimension of solution vector
    # wolves_n  - no. of searching wolves
    # iter_n    - no of iterations
    def optimizer(self,function,max_iter,lb,ub):
        self.gerar_populacao()
        for l in range(0,max_iter):
            for w in self.wolves:
                w.position = np.clip(w.position, lb, ub)
                
                fitness = function(w.position)
                if fitness < self.alpha.obj_score:
                    self.delta.obj_score = self.beta.obj_score  # Update delte
                    self.delta.position = self.beta.position.copy()
                    self.beta.obj_score  = self.alpha.obj_score  # Update beta
                    self.beta.position = self.alpha.position.copy()
                    self.alpha.obj_score = fitness
                    # Update alpha
                    self.alpha.position = w.position.copy()

                if fitness > self.alpha.obj_score and fitness < self.beta.obj_score:
                    self.delta.obj_score = self.beta.obj_score  # Update delte
                    self.delta.position = self.beta.position.copy()
                    self.beta.obj_score = fitness  # Update beta
                    self.beta.position = w.position.copy()

                if fitness > self.alpha.obj_score and fitness > self.beta.obj_score and fitness < self.delta.obj_score:
                    self.delta.obj_score = fitness  # Update delta
                    self.delta.position = w.position.copy()
            a =  2 - l * ((2) / max_iter)
            for w in self.wolves:
                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(self.dim) 
                r2 = np.random.rand(self.dim) 
                
                A1 =2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = abs(C1 * self.alpha.position - w.position) 
                X1 = self.alpha.position - A1 * D_alpha
                
                
                r1 = np.random.rand(self.dim) 
                r2 = np.random.rand(self.dim) 
                
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
    
                D_beta = abs(C2 * self.beta.position - w.position) 
                X2 = self.beta.position - A2 * D_beta
                
                
                r1 = np.random.rand(self.dim) 
                r2 = np.random.rand(self.dim) 
                
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
    
                D_delta = abs(C3 * self.delta.position - w.position) 
                X3 = self.delta.position - A3 * D_delta
                
                w.position = (X1 + X2 + X3)/3
            self.hist.append(self.alpha.obj_score)
        print("global best loss: ", self.alpha.obj_score)
            


        
        return 