import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_pred_nn
from sparse_net_util import sparse_weight, flatten, tf_flatten, unflatten_trainable_variables

# Define a function to compute loss and gradients
def compute_loss_and_gradients(model, tf_x_int, tf_mea, tf_y_obs, tf_input_sb, tf_input_tb, tf_u_sb, tf_u_tb, theta, N_M, alpha, sig_M2, sig_R2, sig_B2):
    with tf.GradientTape(persistent=True) as t:
        with tf.GradientTape(persistent=True) as t_2:
            with tf.GradientTape(persistent=True) as t_1:
                out = model(tf_x_int)
            du_xt = t_1.gradient(out, tf_x_int)
            du_dt = du_xt[:, 1]
            du_dx = du_xt[:, 0]
        du_dxdx = t_2.gradient(du_dx, tf_x_int)[:, 0]

        term1 = -tf.reduce_mean((tf_y_obs - model(tf_mea))**2) / (2 * sig_M2)
        term2 = -tf.reduce_mean((du_dt - theta * du_dxdx)**2) / (2 * sig_R2)
        term3 = -tf.reduce_mean((tf_u_sb - model(tf_input_sb))**2) / (2 * sig_B2)
        term4 = -tf.reduce_mean((tf_u_tb - model(tf_input_tb))**2) / (2 * sig_B2)
        loss = N_M * (term1 + term3 + term4) + (N_M**alpha) * term2

    gradients = t.gradient(loss, model.trainable_variables)
    return loss, gradients, du_dt, du_dxdx

# Define the training function
def train(iterations, hps, data, d_sb, d_tb, inter):
    nn_net = get_pred_nn()  # Initialize main network
    temp_net = get_pred_nn()  # Initialize temporary network for updates

    # Compute the total number of parameters in the network
    pp = len(flatten(nn_net))

    # Calculate N_M (number of measurements) and N_int (number of interior points)
    N_M = data.shape[0]
    N_int = inter.shape[0]

    # Extract hyperparameters
    uu = hps['uu']
    JJ_ratio = hps['JJ_ratio']
    alpha = hps['alpha']
    rho_0 = hps['rho_0']
    rho_1 = hps['rho_1']
    rho = hps['rho']
    init_lr = hps['init_lr']
    delta_prob = hps['delta_prob']
    sig_M2 = hps['sig_M2']
    sig_R2 = hps['sig_R2']
    sig_B2 = hps['sig_B2']

    # Initialize delta, the sparsity mask
    delta = np.random.choice([0, 1], (pp,), p=[1.0 - delta_prob, delta_prob], replace=True)
    JJ = int(JJ_ratio * pp)

    # Lists to store history of parameters and loss
    theta_hist = []
    sparsity_hist = []
    Ws_list = []
    delta_list = []
    theta_hist_keep = []

    lr = init_lr  # Learning rate
    theta = np.random.normal(0, 1)  # Initialize theta parameter

    # Convert boundary and observation data to tensors
    tf_u_sb = tf.convert_to_tensor(d_sb[:, 2:], dtype=tf.float32)
    tf_u_tb = tf.convert_to_tensor(d_tb[:, 2:], dtype=tf.float32)
    tf_mea = tf.stack([data[:, 0], data[:, 1]], axis=1)
    tf_y_obs = tf.convert_to_tensor(data[:, 2:], dtype=tf.float32)
    tf_input_sb = tf.stack([d_sb[:, 0], d_sb[:, 1]], axis=1)
    tf_input_tb = tf.stack([d_tb[:, 0], d_tb[:, 1]], axis=1)
    tf_x_int = tf.stack([inter[:, 0], inter[:, 1]], axis=1)
    tf_x_int = tf.cast(tf_x_int, dtype=tf.float32)
    tf_x_int = tf.Variable(tf_x_int, trainable=True)

    for ii in tqdm(range(1, iterations + 1)):
        # Sparsify the weight to get W_delta
        sparse_weight(nn_net, delta)

        # Compute loss and gradients for the main network
        loss, gradients, du_dt, du_dxdx = compute_loss_and_gradients(
            nn_net, tf_x_int, tf_mea, tf_y_obs, tf_input_sb, tf_input_tb, tf_u_sb, tf_u_tb, theta, N_M, alpha, sig_M2, sig_R2, sig_B2
        )

        # Update theta given W and delta
        temp_X = tf.stack([du_dxdx], 1).numpy()
        temp_Y = tf.stack([du_dt], 1).numpy()

        temp_Sigma0_inv = rho
        temp_SigR2_inv = (N_M**alpha) / (N_int * sig_R2)
        sample_Sig_inv = temp_SigR2_inv * (temp_X.T @ temp_X) + temp_Sigma0_inv
        sample_Sig = 1 / (sample_Sig_inv)
        sample_mu = temp_SigR2_inv * sample_Sig @ (temp_X.T @ temp_Y)

        py_mu = sample_mu.flatten()
        py_Sigma = sample_Sig
        temp_theta = np.random.multivariate_normal(py_mu, py_Sigma)[0]

        # Update W_bar
        flat_gradient = tf_flatten(gradients)
        flat_W = flatten(nn_net)
        delta_comp = 1 - delta
        rand_sample = np.random.normal(0, 1 / np.sqrt(rho_0), len(delta_comp))
        rand_stdnorm = np.random.normal(0, 1, len(delta_comp))
        flat_W += (rand_sample * delta_comp)
        update_term = lr * (-rho_1 * flat_W + flat_gradient) + np.sqrt((2 * lr)) * rand_stdnorm
        update_term *= delta
        flat_W += update_term
        W_bar = flat_W

        # Update delta with W and theta
        aa = uu * np.log(pp) + 0.5 * np.log(rho_0 / rho_1)
        sparsity_hist.append(np.sum(delta) / pp)
        idces = np.random.choice(len(delta), JJ, replace=False)
        delta[idces] = 0
        flat_W_bar_nu = flat_W * delta
        unflatten_trainable_variables(flat_W_bar_nu, temp_net)

        # Compute loss and gradients for the temporary network
        _, temp_grad, _, _ = compute_loss_and_gradients(
            temp_net, tf_x_int, tf_mea, tf_y_obs, tf_input_sb, tf_input_tb, tf_u_sb, tf_u_tb, temp_theta, N_M, alpha, sig_M2, sig_R2, sig_B2
        )

        flat_temp_grad = tf_flatten(temp_grad)
        temp_term = aa + 0.5 * (rho_1 - rho_0) * (flat_W**2) - flat_W * flat_temp_grad - 0.5 * (flat_W**2) * (flat_temp_grad**2)
        temp_qq = 1 / (1 + np.exp(temp_term))
        temp_bin = np.random.binomial(1, temp_qq)
        delta[idces] = temp_bin[idces]

        # Assign updated W to the network
        unflatten_trainable_variables(W_bar, nn_net)
        theta = temp_theta
        theta_hist.append(theta)

        # Periodic logging and visualization
        if ii % (iterations // 10) == 0:
            print(f"Iteration {ii}: Loss = {loss.numpy()}, Theta = {theta}")

        # Store parameters in the last part of training
        if ii >= (iterations - 200000 - 1) and ii % 20 == 0:
            delta_list.append(delta)
            Ws_list.append(W_bar)
            theta_hist_keep.append(theta)

    return Ws_list, theta_hist_keep, delta_list



if __name__ == "__main__":
    from Heat_data_generation import (
        mea_sig, data, d_sb, d_tb, inter
    )

    # Define hyperparameters
    hps = {
        'uu': 1.1,
        'JJ_ratio': 0.02,  # Adjust according to your needs
        'alpha': 1.0,
        'rho_0': data.shape[0],
        'rho_1': 1,
        'rho': 1,
        'init_lr': 1e-10,
        'delta_prob': 1.0,
        'sig_M2': mea_sig**2,
        'sig_R2': 1,
        'sig_B2': 1
    }

    W_sample, theta_sample, delta_sample = train(200000, hps, data, d_sb, d_tb, inter)
    print("Training completed.")
    print("Sampled W:", W_sample)
    print("Sampled theta:", theta_sample)
    print("Sampled delta:", delta_sample)
