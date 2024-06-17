import numpy as np
import tensorflow as tf

def flatten(model):
    """
    Flattens the trainable variables of a given model into a single 1D numpy array.
    
    Args:
        model: A TensorFlow/Keras model.
    
    Returns:
        A 1D numpy array containing all the trainable variables of the model.
    """
    if not model.built:
        # Ensure the model is built by calling it with dummy data
        _ = model(np.zeros((1, 3), dtype=np.float32))
    # Concatenate all trainable variables into a single 1D array
    flattened_vars = np.concatenate([tf.reshape(var, [-1]) for var in model.trainable_variables])
    return flattened_vars

def unflatten_trainable_variables(flattened_vars, model):
    """
    Assigns values from a flattened 1D numpy array back to the model's trainable variables.
    
    Args:
        flattened_vars: A 1D numpy array containing all the trainable variable values.
        model: A TensorFlow/Keras model.
    """
    if not model.built:
        # Ensure the model is built by calling it with dummy data
        _ = model(np.zeros((1, 3), dtype=np.float32))
    # Get the shapes of the trainable variables
    shapes = [var.shape.as_list() for var in model.trainable_variables]
    start = 0
    # Assign values back to the model's trainable variables
    for i, shape in enumerate(shapes):
        size = np.prod(shape)
        flat_var = tf.reshape(flattened_vars[start:start + size], shape)
        flat_var = tf.cast(flat_var, tf.float32)  # Ensure the same data type
        model.trainable_variables[i].assign(flat_var)
        start += size

def tf_flatten(tensors):
    """
    Flattens a list of tensors into a single 1D numpy array.
    
    Args:
        tensors: A list of tensors.
    
    Returns:
        A 1D numpy array containing all the values of the tensors.
    """
    w_dense = []
    # Flatten and concatenate all tensors into a single 1D array
    for tensor in tensors:
        w_dense += tensor.numpy().flatten().tolist()
    return np.array(w_dense)

def sparse_weight(model, delta):
    """
    Applies sparsity to the model's weights by scaling them with a given delta.
    
    Args:
        model: A TensorFlow/Keras model.
        delta: A scalar or array to scale the model's weights.
    """
    # Flatten the model's weights
    flatten_weight = flatten(model)
    # Apply sparsity by scaling the weights
    flatten_weight = flatten_weight * delta
    # Unflatten and assign the scaled weights back to the model
    unflatten_trainable_variables(flatten_weight, model)
