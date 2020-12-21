from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations
import os
from peslearn.constants import package_directory
from peslearn.ml.preprocessing_helper import interatomics_to_fundinvar

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': True, 'morse_alpha': 1.3},
          'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'},
          'scale_y': 'std', 'lr': 0.5}

X, y, Xscaler, yscaler = nn.preprocess(params, nn.raw_X, nn.raw_y)


# How to use 'compute_energy()' function
# --------------------------------------
# E = compute_energy(geom_vectors, cartesian=bool)
# 'geom_vectors' is either:
#  1. A list or tuple of coordinates for a single geometry.
#  2. A column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array:
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#      :       :             :  ],
#  [ coord1, coord2, ..., coordn]]
# In all cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
# If the coordinates format used to train the model was interatomic distances, each set of coordinates should be a 1d array of either interatom distances or cartesian coordinates.
# If cartesian coordinates are supplied, cartesian=True should be passed and it will convert them to interatomic distances.
# The order of coordinates matters. If PES-Learn datasets were used they should be in standard order;
# i.e. cartesians should be supplied in the order x,y,z of most common atoms first, with alphabetical tiebreaker.
# e.g., C2H3O2 --> H1x H1y H1z H2x H2y H2z H3x H3y H3z C1x C1y C1z C2x C2y C2z O1x O1y O1z O2x O2y O2z
# and interatom distances should be the row-wise order of the lower triangle of the interatom distance matrix, with standard order atom axes:
#    H  H  H  C  C  O  O
# H
# H  1
# H  2  3
# C  4  5  6
# C  7  8  9  10
# O  11 12 13 14 15
# O  16 17 18 19 20 21

# The returned energy array is a column vector of corresponding energies. Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.


def pes(geom_vectors, cartesian=True):
    model = torch.load('model.pt')
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    morse_grad = transform_x_buffer(g, params)
    morse_grad_array = np.array(morse_grad)
    print("The gradient computed by preprocessing: ", morse_grad)
    # Divide the gradient computation into different sub-categories
    x_scale = Xscaler.scale_
    if "mm" not in params["scale_X"]["scale_X"]:
        x_scale = 1 / x_scale
    print("The scale of x transformation: ", x_scale)
    concatenated_gradient = []
    for i in range(len(x_scale)):
        x_scale_element = x_scale[i]
        morse_grad_element = morse_grad_array[i]
        concatenated_gradient.append(x_scale_element * morse_grad_element)
    concatenated_gradient_array = np.array(concatenated_gradient)
    x = torch.tensor(data=newX, requires_grad=True)
    model.zero_grad()
    # x = torch.tensor(data=x)
    E = model(x.float())
    v = torch.tensor([[1.0]], dtype=torch.double)
    E.backward(v)
    # Start from the computation of gradient with the scaling of y
    # Change the file to the pytorch code
    energy_to_compute = E.detach()
    e = nn.inverse_transform_new_y(energy_to_compute, yscaler)
    # The scale is reversed after reversion. No need to reverse for y scaling.
    y_scale = yscaler.scale_
    if "mm" in params["scale_y"]:
        y_scale = 1 / y_scale
    print("Before gradient: ", x.grad)
    main_grad = np.array(x.grad * y_scale)
    print("Main grad: ", main_grad[0])
    final_grad = []
    for i in range(len(main_grad[0])):
        grad_element = main_grad[0][i] * concatenated_gradient[i]
        final_grad.append(grad_element)
    final_result = np.asarray(final_grad)
    # e = e - (insert min energy here)
    # e *= 219474.63  (convert units)
    print(final_result)
    return e, final_result


def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1, 3)
    n = len(vec)
    distance_matrix = np.zeros((n, n))
    for i, j in combinations(range(len(vec)), 2):
        R = np.linalg.norm(vec[i] - vec[j])
        distance_matrix[j, i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix), -1)]
    return distance_vector


# This function substitute the original preprocessing function to
def transform_x_buffer(cartesian_coordinate, parameters):
    x_tensor = torch.tensor(cartesian_coordinate, requires_grad=True)
    if len(cartesian_coordinate.shape) > 2:
        raise ValueError("The input dimension goes beyond the valid region.")
    x_tensor_after_transformation = x_tensor
    if parameters["morse_transform"]["morse"]:
        alpha = parameters["morse_transform"]["morse_alpha"]
        x_tensor_after_transformation = torch.exp(-x_tensor / alpha)
    # if parameters["pip"]["pip"]:
    #     raise ValueError("Unable to compute the gradient passing to the pip.")
    x_sum = x_tensor_after_transformation.sum()
    x_sum.backward()
    return x_tensor.grad


# Changed to new branch luoshu_implementation
if __name__ == "__main__":
    print("Start the calculation of energy...")
    # Based on the data in PES_data_new 17
    input_value = (0, 0, 1.1125, 0, 0.85, -0.12, 0, 0, 0)
    result, grad = pes(geom_vectors=input_value, cartesian=True)
    print("Computed energy: ", result)
    print("Derived gradient: ", grad)
