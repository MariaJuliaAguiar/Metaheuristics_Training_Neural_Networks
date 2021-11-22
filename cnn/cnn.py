import numpy as np
from cnn.conv_layer import ConvLayer, MaxPoolLayer,FullyConnectedLayer

class ConvolutionalNN:
     def __init__(self,X,Y):
        self.X = X
       
        self.Y = Y
        self.layers = []
        self.n_categories = np.shape(self.Y)[1]# numero de classes
        self.n_images = np.shape(self.Y)[0] # número de imagens de entrada
    
     def add_conv_layer(self, n_filters, kernel_size, eta=0.001):
        self.layers.append(ConvLayer(n_filters, kernel_size, eta))
     
     def add_maxpool_layer(self, stride=2):
        self.layers.append(MaxPoolLayer(stride))
        
     def add_fullyconnected_layer(self, eta=0.1):
        self.layers.append(FullyConnectedLayer(self.n_categories, self.n_images, eta))
           # Define data for training or testing
     def new_input(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
        self.n_images = np.shape(self.Y)[0]
        self.layers[-1].update_images(self.n_images)
    # Forward propagation through all layers
     def forward_propagation(self,p):
        input = self.X
        
        for layer in self.layers:
            new_input = layer.feed_forward(input,self.Y,p)
            input = new_input

        return new_input
     def predict(self, p, X_test):
        print("Entrei aqui")
        input = X_test
        for seq in range(4):
            new_input = self.layers[seq].feed_forward(input, self.Y,p)
            input = new_input
        pred = self.layers[4].predict(p, input)
        print("Cheguei aqui")
        return pred