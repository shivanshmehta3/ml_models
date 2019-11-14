import numpy as np
class NeuralNetwork:
    @staticmethod
    def computeCost(nn_params, input_layer_size, hidden_layer_size, output_layer_size, X, y, lda):
        # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
        # for our 2 layer neural network
        Theta1 = np.resize( nn_params[0:(hidden_layer_size * (input_layer_size+1)) ], (hidden_layer_size, (input_layer_size + 1)) )
        Theta2 = np.resize( nn_params[(hidden_layer_size * (input_layer_size + 1)):], (output_layer_size, (hidden_layer_size + 1)))

        [m,n] = X.shape
        J = 0
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)

        a1 = np.concatenate((np.ones((m, 1)), X),axis=1)
        z2 = a1.dot(Theta1.T)
        a2 = NeuralNetwork.sigmoid(z2)
        a2 = np.concatenate((np.ones((m, 1)), a2),1)
        z3 = a2.dot(Theta2.T)
        hypothesis_values = NeuralNetwork.sigmoid(z3)
        log_hypothesis = np.log(hypothesis_values)
        log_one_minus_hypothesis = np.log(1 - hypothesis_values)
        y_temp = np.zeros((m,output_layer_size))
        iter_arr = np.linspace(0,m-1,m, dtype=int)
        for iter in iter_arr:
            y_temp[iter, (y[iter] - 1)] = 1
        theta1_without_first_parameter = Theta1
        theta1_without_first_parameter[:,0] = 0
        theta2_without_first_parameter = Theta2
        theta2_without_first_parameter[:,0] = 0
        theta_without_first_parameter = np.concatenate((np.resize(theta1_without_first_parameter,(theta1_without_first_parameter.size,1)), np.resize(theta2_without_first_parameter,(theta2_without_first_parameter.size,1))))
        J_fn_regularization_term = (lda/(2*m) * sum(theta_without_first_parameter * theta_without_first_parameter))
        J = (-(1/m) * sum(sum((y_temp * log_hypothesis) + ((1 - y_temp) * log_one_minus_hypothesis))) + J_fn_regularization_term)
        
        Delta1 = np.zeros(Theta1_grad.shape)
        Delta2 = np.zeros(Theta2_grad.shape)
        for iter in iter_arr:
            a1=np.concatenate(([1],X[iter,:]))
            a1 = np.resize(a1, (1, a1.size))
            z2 = a1.dot(Theta1.T)
            a2 = NeuralNetwork.sigmoid(z2)
            a2=np.concatenate(([1],a2[0,:]))
            a2 = np.resize(a2, (1, a2.size))
            z3 = a2.dot(Theta2.T)
            a3 = NeuralNetwork.sigmoid(z3)
            delta3 = a3 - y_temp[iter,:]
            delta2 = (delta3 @ Theta2) * (a2*(1-a2))  # `@` is cross product
            [delta2_rows,delta2_cols] = delta2.shape
            delta2 = delta2[:, 1: delta2_cols]
            Delta1 = Delta1 + (delta2.T @ a1)
            Delta2 = Delta2 + (delta3.T @ a2)
        Theta1_grad = (1/m)*(Delta1 + (lda*Theta1))
        Theta1_grad[:,0] = (1/m)*Delta1[:,0]
        Theta2_grad = (1/m)*(Delta2 + (lda*Theta2))
        Theta2_grad[:,0] = (1/m)*Delta2[:,0]

        #Unroll gradients
        grad = np.concatenate((np.resize(Theta1_grad, (Theta1_grad.size,1)), np.resize(Theta2_grad, (Theta2_grad.size,1))))

        return J, grad

    @staticmethod
    def sigmoid(z):
        g = 1.0 / (1.0 + np.exp(-1 * z))
        return g

    @staticmethod
    def sigmoidGradient(z):
        [z_rows,z_cols] = z.shape
        g = np.zeros((z_rows,1))
        g = NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))
        return g

    @staticmethod
    def randInitializeWeights(L_in, L_out):
        W = np.zeros((L_out, 1 + L_in))
        epsilon_init = 0.12
        W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init
        return W