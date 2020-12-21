from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': True, 'morse_alpha': 1.3},
          'pip': {'degree_reduction': False, 'pip': False},
          'scale_X': {'activation': 'tanh', 'scale_X': 'std'}, 'scale_y': 'std', 'lr': 0.5}

# Why all r1 in the input data is 0.97? I am very confused.
X, y, Xscaler, yscaler = nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')


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

# The returned energy array is a column vector of corresponding energies.
# Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.

# TODO: Last step of implementation: do the backpropagation of pip (fundamental to invariant)
def pes(geom_vectors, cartesian=True):
    model = torch.load('model.pt')
    g = np.asarray(geom_vectors)
    gradients_wrt_cartesian_coordinates = np.zeros((3, 3), dtype=float)
    g_test_plus = g + np.array([0, 1e-2, 0, 0, 0, 0, 0, 0, 0])
    g_test_minus = g - np.array([0, 1e-2, 0, 0, 0, 0, 0, 0, 0])
    g_test_plus_new = np.expand_dims(g_test_plus, axis=0)
    g_test_minus_new = np.expand_dims(g_test_minus, axis=0)
    internal_coordinate_plus = np.apply_along_axis(cart1d_to_distances1d, 1, g_test_plus_new)
    internal_coordinate_minus = np.apply_along_axis(cart1d_to_distances1d, 1, g_test_minus_new)
    g_test = (internal_coordinate_plus[0] - internal_coordinate_minus[0])/np.array([1e-2, 1e-2, 1e-2])
    internal_coordinate_real = np.apply_along_axis(cart1d_to_distances1d, 0, g)
    print("Numerical gradient for Cartesian conversion: ", g_test)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        gradients_wrt_cartesian_coordinates = np.apply_along_axis(cart_conversion_gradient, axis, g)
        print("Analytical gradient for Cartesian conversion: ", gradients_wrt_cartesian_coordinates)
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    morse_grad = transform_x_buffer(g, params)
    morse_grad_array = np.array(morse_grad)
    print("The gradient computed by morse preprocessing: ", morse_grad)
    # Divide the gradient computation into different sub-categories
    x_scale = Xscaler.scale_
    # Because the scale returned by the std scalar is reversed (input wrt output), need to reverse the result to
    # balance.
    if "mm" not in params["scale_X"]["scale_X"]:
        x_scale = 1 / x_scale
    # On the other hand, the scale of MinMaxScaler is not reversed (output wrt input), no need to reverse.
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
    E.backward()
    print(E.grad_fn)
    print("Gradient of x: ", x.grad)
    # Start from the computation of gradient with the scaling of y
    # Change the file to the pytorch code
    energy_to_compute = E.detach()
    e = nn.inverse_transform_new_y(energy_to_compute, yscaler)
    # The scale is reversed after reversion. No need to reverse for y scaling.
    y_scale = yscaler.scale_
    if "mm" in params["scale_y"]:
        y_scale = 1 / y_scale
    nn_grad = x.grad
    nn_grad.require_grad_ = False
    print("Before gradient: ", nn_grad)
    main_grad = np.array(nn_grad * y_scale)
    print("Main grad: ", main_grad[0])
    final_grad = []
    for i in range(len(main_grad[0])):
        grad_element = main_grad[0][i] * concatenated_gradient_array[i]
        final_grad.append(grad_element)
    final_result = gradients_wrt_cartesian_coordinates.dot(np.array(final_grad))
    # e = e - (insert min energy here)
    # e *= 219474.63  (convert units)
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


# This function copies the function of conversion of cartesian coordinate to the distance for the purpose of gradient
# computation.
# TODO: verify the result of gradient computation in Cartesian coordinates conversion.
def cart_conversion_gradient(coordinates):
    coordinates_tensor = torch.tensor(coordinates, requires_grad=True)
    coordinate_reorganized = torch.reshape(coordinates_tensor, (len(coordinates_tensor)//3, 3))
    # The internal coordinates are saved in this matrix.
    # internal_coordinates_matrix = torch.zeros((len(coordinate_reorganized), len(coordinate_reorganized)))
    # for i, j in combinations(range(len(coordinate_reorganized)), 2):
        # norm = torch.norm(coordinate_reorganized[i] - coordinate_reorganized[j])
        # internal_coordinates_matrix[j, i] = norm
    # internal_coordinates_vector = internal_coordinates_matrix[np.tril_indices(len(internal_coordinates_matrix), -1)]
    gradients = []
    for i in range(len(coordinates)//3):
        # internal_coordinates_vector[0].backward(retain_graph=True)
        # gradients.append(list(np.array(coordinates_tensor.grad)))
        gradients.append(get_gradient_from_cart_conversion(coordinates, i))
    gradient_matrix = np.array(gradients)
    return gradient_matrix.transpose()


def get_gradient_from_cart_conversion(coordinates_file, target_axis):
    coordinates_vector = torch.tensor(coordinates_file, requires_grad=True)
    coordinates_matrix = torch.reshape(coordinates_vector, (len(coordinates_vector)//3, 3))
    internal_coordinates_matrix = torch.zeros((len(coordinates_matrix), len(coordinates_matrix)))
    for i, j in combinations(range(len(coordinates_matrix)), 2):
        norm = torch.norm(coordinates_matrix[i] - coordinates_matrix[j])
        internal_coordinates_matrix[j, i] = norm
    internal_coordinates_vector = internal_coordinates_matrix[np.tril_indices(len(internal_coordinates_matrix), -1)]
    internal_coordinates_vector[target_axis].backward()
    target_gradient = np.array(list(np.array(coordinates_vector.grad)))
    return target_gradient


# This function copies the original preprocessing function for the purpose of gradient computation.
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
