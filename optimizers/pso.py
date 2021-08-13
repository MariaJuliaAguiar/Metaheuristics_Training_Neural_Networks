import numpy as np
import random
import pandas as pd

class Particle:
    def __init__(self,n_particles,dim):
        
        self.pbest = np.inf
        self.pbestpos = np.zeros((dim,))
        
        self.vel = np.zeros((dim,))
        self.positions = np.zeros((dim,)) #lista referente a populacao corrent
    
class PSO:
    def __init__(self,n_particles,dim,Vmax,wMax,wMin,c1,c2,initpos):
        self.gbest = np.inf
        self.gbestpos = np.zeros((dim,))
        
        self.Vmax_m  = Vmax
        self.wMax_m = wMax
        self.wMin_m  = wMin
        self.c1_m  = c1
        self.c2_m  = c2
        self.Particle = np.array(
            [Particle(n_particles,dim) for i in range(n_particles)]
        )
        self.hist=[]
        self.initpos = initpos
    def gerar_populacao(self):
        i=0
        for particle in self.Particle:
            particle.positions = self.initpos[i]
            i=i+1
        return
    
    
    def optimizer(self,function,max_iter,lb,ub):
        self.gerar_populacao()
        for l in range(max_iter):
            for particle in self.Particle:
                particle.positions = np.clip(particle.positions, lb, ub)
                
                fitness = function(particle.positions)
                #print(fitness)
                if particle.pbest > fitness:
                    particle.pbest = fitness
                    particle.pbestpos = particle.positions.copy()

                if self.gbest > fitness:
                    self.gbest = fitness
                    self.gbestpos = particle.positions.copy()
            
            w = self.wMax_m - l * ((self.wMax_m - self.wMin_m) / max_iter)
            for particle in self.Particle:
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                particle.vel = (w * particle.vel) + (self.c1_m * r1 * (particle.pbestpos - particle.positions)) + (self.c2_m * r2 * (self.gbestpos - particle.positions))
                particle.vel = np.clip(particle.vel,  -self.Vmax_m, self.Vmax_m)
                #if particle.vel> self.Vmax_m:
                    #particle.vel = self.Vmax_m

                #if particle.vel < -self.Vmax_m:
                    #particle.vel = -self.Vmax_m

                particle.positions = particle.positions + particle.vel
            self.hist.append(self.gbest)
        print("global best loss: ", self.gbest)
        return          