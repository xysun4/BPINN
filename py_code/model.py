import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

def get_pred_nn(hsize=[64, 64, 64, 64]):
    """
    Builds and returns a sequential neural network model with specified hidden layer sizes.

    Args:
        hsize (list of int): List of integers representing the number of units in each hidden layer.

    Returns:
        tensorflow.keras.models.Sequential: A Keras sequential model.
    """
    # Define the initializer (GlorotUniform in this case)
    initializer = GlorotUniform()
    
    # Create a Sequential model
    model = Sequential()

    # Add the first hidden layer with input shape
    model.add(Dense(hsize[0], input_shape=(2,), kernel_initializer=initializer))
    model.add(Activation('tanh'))

    # Add subsequent hidden layers
    for units in hsize[1:]:
        model.add(Dense(units, kernel_initializer=initializer))
        model.add(Activation('tanh'))

    # Add the output layer
    model.add(Dense(1, activation='linear'))
    
    return model
