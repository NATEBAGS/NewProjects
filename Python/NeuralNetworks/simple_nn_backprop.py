import numpy as np 

# defining the backprop for Sigmoid Function
def back_sigmoid(x):
  return x*(1-x)

# defining the Sigmoid Function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def forward_pass(X, weights_input_hidden, weights_hidden_output):

    # pre-activation of the hidden input layer (a_2, a_3, a_4)
    pre_act_inp = X @ weights_input_hidden

    # activations z_2, z_3, z_4
    post_act_inp = sigmoid(pre_act_inp)

    # pre-activation of the hidden output layer
    pre_act_out = post_act_inp @ weights_hidden_output

    # activation of z_1
    post_act_out = sigmoid(pre_act_out)
    
    ## output is of shape (N, 1)
    ## hiddenLayer_activations is of shape (N, 3)
    return post_act_out, post_act_inp


def backward_pass(X, y, output, weights_hidden_output, weights_input_hidden, hiddenLayer_activations):
  # calculating rate of change of error w.r.t weight between hidden and output layer

    N = X.shape[0]
    # Outputs what δ_1 is as the output neuron
    output_delta = (output - y) * back_sigmoid(output)

    # Gradient wrt hidden layer to output layer weights
    error_wrt_weights_hidden_output = hiddenLayer_activations.T @ output_delta / N

    # Delta at the hidden layer (δ_2, δ_3, δ_4)
    hidden_delta = (output_delta @ weights_hidden_output.T) * back_sigmoid(hiddenLayer_activations)
    
    # Average gradient for each weight matrix
    error_wrt_weights_input_hidden = X.T @ hidden_delta / N
    
    return error_wrt_weights_hidden_output, error_wrt_weights_input_hidden

def train(X_train, y_train):

    # defining the model architecture
    inputLayer_neurons = 1  # number of neurons at input
    hiddenLayer_neurons = 3  # number of hidden layers neurons
    outputLayer_neurons = 1  # number of neurons at output layer

    # initializing weight
    weights_input_hidden = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
    weights_hidden_output = np.random.uniform(
        size=(hiddenLayer_neurons, outputLayer_neurons)
    )
    # defining the parameters
    lr = 0.1
    epochs = 1000

    losses =  []

    for ep in range(epochs):
      output_, hiddenLayer_activations = forward_pass(X_train, weights_input_hidden, weights_hidden_output)

      ## Backward Propagation
      # calculating error
      error = np.square(y_train - output_) / 2

      error_wrt_weights_hidden_output, error_wrt_weights_input_hidden = backward_pass(X_train, y_train, 
                                                                                      output_, 
                                                                                      weights_hidden_output, 
                                                                                      weights_input_hidden, 
                                                                                      hiddenLayer_activations)

      # updating the weights
      weights_hidden_output = weights_hidden_output - lr * error_wrt_weights_hidden_output
      weights_input_hidden = weights_input_hidden - lr * error_wrt_weights_input_hidden

      # print error at every 100th epoch
      epoch_loss = np.average(error)
      if ep % 100 == 0:
          print(f"Error at epoch {ep} is {epoch_loss:.5f}")

      # appending the error of each epoch
      losses.append(epoch_loss)

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    final_pred, _ = forward_pass(X_train, weights_input_hidden, weights_hidden_output)

    plt.plot(final_pred, c='r')
    plt.plot(y_train, c='b')
    plt.show()

    print("Weights of the network are: ")
    print("w13, w12, w14", weights_hidden_output.T)
    print("w35, w25, w45", weights_input_hidden)


# defining training data
X_train = np.zeros((11, 1)).astype(np.float32)
y_train = np.zeros((11, 1)).astype(np.float32)

X_train[0] = -3
X_train[1] = -2
X_train[2] = -1.5
X_train[3] = -1.0
X_train[4] = -0.5
X_train[5] = 0.0
X_train[6] = 0.5
X_train[7] = 1.0
X_train[8] = 1.5
X_train[9] = 2.0
X_train[10] = 3.0


y_train[0] = 0.7312
y_train[1] = 0.7339
y_train[2] = 0.7438
y_train[3] = 0.7832
y_train[4] = 0.8903
y_train[5] = 0.9820
y_train[6] = 0.8114
y_train[7] = 0.5937
y_train[8] = 0.5219
y_train[9] = 0.5049
y_train[10] = 0.5002

train(X_train, y_train)
