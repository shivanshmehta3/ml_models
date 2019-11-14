import nn_model.NeuralNetwork as nn
import nn_model.config as cfg
import numpy as np
import scipy.optimize as op
import data.debug_data as data

#load debug data

initial_Theta1 = nn.NeuralNetwork.randInitializeWeights(cfg.INPUT_LAYER_SIZE, cfg.HIDDEN_LAYER_SIZE)
initial_Theta2 = nn.NeuralNetwork.randInitializeWeights(cfg.HIDDEN_LAYER_SIZE, cfg.OUTPUT_LAYER_SIZE)

initial_nn_params = np.concatenate((np.resize(initial_Theta1, (initial_Theta1.size,1)), np.resize(initial_Theta2, (initial_Theta2.size,1))))

# Theta1 = data.Theta1
# Theta2 = data.Theta2
# initial_nn_params = np.concatenate((np.resize(Theta1, (Theta1.size,1)), np.resize(Theta2, (Theta2.size,1))))
lda = 3
[j,grad]=nn.NeuralNetwork.computeCost(initial_nn_params, cfg.INPUT_LAYER_SIZE, cfg.HIDDEN_LAYER_SIZE, cfg.OUTPUT_LAYER_SIZE, data.X, data.y,lda)
print(j)
print(nn.NeuralNetwork.sigmoidGradient(np.array([[-1,0]])))