import os
import numpy as np
import my_math


def print_new_section():
    print("================================================\r\n\r\n")


current_path = os.path.dirname(os.path.abspath(__file__))
current_problem = "transportation"
current_problem_path = "{}/problems/{}/".format(current_path, current_problem)
np_options_arr, original_options_with_text, original = my_math.read_matrix(
    "{}{}".format(current_problem_path, "options.txt"))

print_new_section()
print("options \r\n{}".format(original_options_with_text))

np_criteria_arr, np_criteria_original_with_text, original = my_math.read_matrix(
    "{}{}".format(current_problem_path, "criteria.txt"))

print_new_section()
print("criteria \r\n{}".format(np_criteria_arr))

size_y = np_criteria_arr.shape[1]
size_x = np_criteria_arr.shape[0]

np_importance_arr, original_with_text, original_importance = my_math.read_matrix(
    "{}{}".format(current_problem_path, "importance.txt"))

print("importance \r\n{}".format(np_importance_arr))

importance_array = my_math.generate_importance_correlation_matrix(original_importance)

print_new_section()
print("importance correlation \r\n{}".format(importance_array))

eigens = np.linalg.eig(importance_array)
print("eigens values \r\n{}".format(eigens[0].real))
print("eigens vectors \r\n{}".format(eigens[1].real))

eigens_values = eigens[0]
max_eigen_value_index = eigens_values.argmax()
print_new_section()
print("max eigen value {}".format(eigens_values.max()))
print("max eigen value index \r\n{}".format(max_eigen_value_index))

eigens_vectors = eigens[1]
selected_eigen_vector = eigens_vectors[max_eigen_value_index]
print_new_section()
print("Selected eigen vector \r\n{}".format(selected_eigen_vector))

for ei in range(selected_eigen_vector.size):
    eigen_row_value = selected_eigen_vector[ei]
    for ai in range(np_criteria_arr[ei].size):
        np_criteria_arr[ei][ai] = np_criteria_arr[ei][ai] ** ei

print_new_section()
print("options ^ eigen vector \r\n{}".format(np_criteria_arr))
arr_result = []
for hi in range(size_x):  # horizontal
    min_value = 1
    for vi in range(size_y):
        value = np_criteria_arr[hi][vi]
        if value < min_value:
            min_value = value
    arr_result.append(min_value)

print_new_section()
print("min options \r\n{}".format(arr_result))

max_value = 0
max_value_index = 0
for i in range(size_y):
    value = arr_result[i]
    if value > max_value:
        max_value = value

print_new_section()
print("Solution \r\n{} with {}".format(original_options_with_text[max_value_index], max_value))
