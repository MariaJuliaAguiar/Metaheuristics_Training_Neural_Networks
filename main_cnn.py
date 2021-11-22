import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import transform_targets
from cnn.cnn import ConvolutionalNN
from cnn.conv_layer import ConvLayer, MaxPoolLayer,FullyConnectedLayer
from optimizers.pso import PSO

print("Importando dados")
digits  = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

##Training and Testing
X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
#model
print("Criando modelo")

model = ConvolutionalNN(X_train[0:100, :], Y_train[0:100, :])
#modelo  foi configurada com duas camadas convolucionais com 10 e 2 filtros respectivamente de tamanho 2x2
model.add_conv_layer(n_filters=5, kernel_size=[2, 2])
# camada de max pooling de stride =2
model.add_maxpool_layer(stride=2)
model.add_conv_layer(n_filters=2, kernel_size=[2, 2])
model.add_maxpool_layer(stride=2)
model.add_fullyconnected_layer()


n_images = np.shape(X_train)[0]
indices = np.arange(0, n_images)
batch = 10

r_indices = np.random.choice(indices, size=batch)
model.new_input(X_train[r_indices, :], Y_train[r_indices, :])

#initpos = np.random.randn(100,410)
initpos = np.random.uniform(low=-2, high=2, size=(100,410))

#pred = model.forward_propagation(initpos)
#PSO Hyperparameters

c1 = 0.9
c2 = 0.3
Wmin = 0.9
wMax = 0.9
Vmax = 6
population_size = 100
lb = [-2]
up = [2]
dimensions = 410
number_iterations=200
hist_sim_pso = []
best_sim_pso = []
best_pso = []
print("Otimizando")
pso = PSO(population_size,dimensions,Vmax,wMax,Wmin,c1,c2,initpos)
pso.optimizer(model.forward_propagation, number_iterations,lb,up)
hist_sim_pso.append(pso.hist)
best_sim_pso.append(pso.gbestpos)
best_pso.append(pso.gbest)
 # Convergence curves - Plots
numpy_array_PSO = np.array(hist_sim_pso)
transpose_PSO = numpy_array_PSO.T
    # Plot current gBest
results_path = "./results/"

matplotlib.use('Agg')
plt.plot(transpose_PSO)
plt.xlabel("Iteration")
plt.ylabel("gBest acc")
plt.savefig(results_path + "PSO_convergence"  + ".png")
plt.close()
p = best_sim_pso
n_images1 = np.shape(X_train)[0]
indices1 = np.arange(0, n_images1)
r_indices1 = np.random.choice(indices1, size=batch)

train =  np.argmax(Y_train[r_indices1,:], axis=1)

acc_PSO =(model.predict(p,X_train[r_indices1, :])==train).mean()
train_predict = model.predict(p,X_train[r_indices1, :])
min_ind_PSO = np.argmin(best_pso)
print("PSO Accuracy: " + str(acc_PSO))
print("PSO Best fitness:" + str(best_pso[min_ind_PSO]))

n_images1 = np.shape(X_test)[0]
indices1 = np.arange(0, n_images1)
batch = 10

r_indices1 = np.random.choice(indices1, size=batch)
teste = np.argmax(Y_test[r_indices1,:], axis=1)
acc_PSO_teste =(model.predict(p,X_test[r_indices1, :])==teste).mean()
teste_predict = model.predict(p,X_test[r_indices1, :])

print("PSO Accuracy test: " + str(acc_PSO_teste))
print("PSO Best fitness:" + str(best_pso[min_ind_PSO]))