import numpy as np

class NeuralNetwork:
    def __init__(self,X,Y,input_nodes,hidden_nodes,output_nodes):
        self.X = X
       
        self.Y = Y
        
        self.n_inputs = input_nodes
        self.n_hidden = hidden_nodes
        self.n_classes = output_nodes
        self.num_samples = X.shape[0] 
        
    def logits_function(self,p,X_train):
        """ Calculate roll-back the weights and biases
    
        Inputs
        ------
        p: np.ndarray
            The dimensions should include an unrolled version of the
            weights and biases.
    
        Returns
        -------
        numpy.ndarray of logits for layer 2
        
    
        """
        #coeficientes
        c1 = self.n_hidden * self.n_inputs
        c2 = c1 + self.n_hidden
        c3 = c2 + (self.n_hidden * self.n_hidden)
        c4=  c3 + self.n_hidden
        c5 = c4+(self.n_hidden*self.n_classes)
        
         # First layer weights
        W1 = p[0:c1].reshape(self.n_inputs, self.n_hidden)
        #print(W1)
        # First layer bias
        b1 = p[c1:c2].reshape(( self.n_hidden,))
        
        # Second layer weights
        W2 = p[c2:c3].reshape(self.n_hidden, self.n_hidden) 
        
        # Second layer bias
        b2 = p[c3:c4].reshape(( self.n_hidden,))
        W3 = p[c4:c5].reshape(self.n_hidden, self.n_classes) 
        b3 = p[c5: c5 + self.n_classes].reshape((self.n_classes,))
        
    
        # Perform forward propagation
        z1 = X_train.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        
        # Second linear step
        z2 = a1.dot(W2) + b2
        
        # Second activation function
        a2 = np.tanh(z2)
        
        #Third linear step
        z3 = a2.dot(W3) + b3
        
        #For the Third linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer
        logits = z3
        
        return logits          # Logits for Layer 2
   
    # Forward propagation
    def forward_prop(self,params):
        """Forward propagation as objective function
    
        This computes for the forward propagation of the neural network, as
        well as the loss.
    
        Inputs
        ------
        params: np.ndarray
            The dimensions should include an unrolled version of the
            weights and biases.
    
        Returns
        -------
        float
            The computed negative log-likelihood loss given the parameters
        """
    
        logits = self.logits_function(params,self.X)
    
        # Compute for the softmax of the logits
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
        # Compute for the negative log likelihood
    
        corect_logprobs = -np.log(probs[range(self.num_samples), self.Y])
        loss = np.sum(corect_logprobs) / self.num_samples
        #print(loss)
        return np.array(loss)
    def predict(self,pos,X_test):
        """
        Use the trained weights to perform class predictions.
    
        Inputs
        ------
        pos: numpy.ndarray
            Position matrix found by the swarm. Will be rolled
            into weights and biases.
        """
        logits = self.logits_function(pos,X_test)
        
        y_pred = np.argmax(logits, axis=1)
       
        return y_pred
    