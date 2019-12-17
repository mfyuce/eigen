import os
import enum
import numpy as np
import my_math


class Importance(enum.Enum):
    """
    (1,1,1) =>  0 =>  Same
    (2/3,1,3/2) =>  1 =>  Match
    (3/2,2,5/2) =>  2 =>  Important
    (5/2,3,7/2) =>  3 =>  MoreImportant
    (7/2,4,9/2) =>  4 =>  CertainlyImportant
    """
    Same = 0
    Match = 1
    Important = 2
    MoreImportant = 3
    CertainlyImportant = 4


ImportanceValues = {
    Importance.Same.name: np.array([1, 1, 1]),
    Importance.Match.name: np.array([2 / 3, 1, 3 / 2]),
    Importance.Important.name: np.array([3 / 2, 2, 5 / 2]),
    Importance.MoreImportant.name: np.array([5 / 2, 3, 7 / 2]),
    Importance.CertainlyImportant.name: np.array([7 / 2, 4, 9 / 2])
}

size_lmu = 3
LMU_l_INDEX = 0
LMU_M_INDEX = 1
LMU_U_INDEX = 2


def print_new_section():
    print("================================================\r\n\r\n")


current_path = os.path.dirname(os.path.abspath(__file__))
current_problem = "ahp"
current_problem_path = "{}/problems/{}/".format(current_path, current_problem)
np_options_arr, original_options_with_text, original = my_math.read_matrix(
    "{}{}".format(current_problem_path, "options.txt"))

print_new_section()
print("options \r\n{}".format(original_options_with_text))

np_criteria_parent_arr, np_criteria_parent_original_with_text, original_parent = my_math.read_matrix(
    "{}{}".format(current_problem_path, "criteria_parent.txt"))

print_new_section()
print("criteria \r\n{}".format(np_criteria_parent_original_with_text))

np_importance_parent_arr, original_importance_parent_with_text, original_importance_parent = my_math.read_matrix(
    "{}{}".format(current_problem_path, "importance_parent.txt"))

print("importance \r\n{}".format(original_importance_parent_with_text))


def find_criteria_index(criteria, criteria_list):
    index_ret = -1
    for criteria_list in criteria_list:
        index_ret = index_ret + 1
        criteria_name = criteria_list[0]
        if criteria_name == criteria:
            return index_ret


def symetric_inverse_single_dim(a):
    len = a.shape[0]
    arr_ret = np.zeros(len)
    for i in range(len):
        arr_ret[len - i - 1] = a[i]
    return arr_ret


def get_weighted_compare_matrix(np_criteria_parent_original_with_text, original_importance_parent_with_text):
    size_parent = len(np_criteria_parent_original_with_text)
    importance_parent = np.zeros((size_parent, size_parent), dtype=np.object)
    importance_parent = importance_parent.reshape(size_parent, size_parent)

    # Complete inferior part with inverse values 1/x
    for importance_list in original_importance_parent_with_text:
        importance = Importance[importance_list[0]]
        importance_value = ImportanceValues[importance.name]
        importance_name_a = importance_list[1]
        importance_name_b = importance_list[2]
        importance_index_a = find_criteria_index(importance_name_a, np_criteria_parent_original_with_text)
        importance_index_b = find_criteria_index(importance_name_b, np_criteria_parent_original_with_text)
        importance_parent[importance_index_a][importance_index_b] = importance_value
        if isinstance(importance_parent[importance_index_b][importance_index_a], int) :
            importance_parent[importance_index_b][importance_index_a] = symetric_inverse_single_dim(
                1 / importance_value)

    # set diagonal to identity / same
    for a in range(size_parent):
        importance_parent[a][a] = ImportanceValues[Importance.Same.name]

    # if any remaining zeros
    for i in range(size_parent):
        for j in range(size_parent):
            if isinstance(importance_parent[i][j], int):
                m = importance_parent[j][i]
                if m == 0:
                    pass
                importance_parent[i][j] = symetric_inverse_single_dim(1 / m)

    lmu = np.zeros((size_parent, size_lmu))
    for c in range(size_parent):
        for d in range(size_lmu):
            for a in range(size_parent):
                lmu[c][d] = lmu[c][d] + importance_parent[c][a][d]

    sum_lmu = np.sum(lmu, axis=0)

    s_lmu = np.zeros((size_parent, size_lmu))

    for c in range(size_parent):
        for d in range(size_lmu):
            s_lmu[c][d] = lmu[c][d] / sum_lmu[size_lmu - d - 1]

    # compare matrix
    compare_matrix_parent = np.zeros((size_parent, size_parent))

    for i in range(size_parent):
        for j in range(size_parent):
            if s_lmu[i][LMU_M_INDEX] > s_lmu[j][LMU_M_INDEX]:
                compare_matrix_parent[i][j] = 1
            elif s_lmu[i][LMU_U_INDEX] < s_lmu[j][LMU_l_INDEX]:
                compare_matrix_parent[i][j] = 0
            else:
                compare_matrix_parent[i][j] = (s_lmu[j][LMU_l_INDEX] - s_lmu[i][LMU_U_INDEX]) \
                                              / ((s_lmu[j][LMU_l_INDEX] - s_lmu[j][LMU_M_INDEX])
                                                 - (s_lmu[i][LMU_U_INDEX] - s_lmu[i][LMU_M_INDEX]))

    min_compare_matrix_parent = np.min(compare_matrix_parent, axis=1)
    sum_min_compare_matrix_parent = np.sum(min_compare_matrix_parent)
    weighted_min_compare_matrix_parent = np.zeros(size_parent)
    for i in range(size_parent):
        weighted_min_compare_matrix_parent[i] = min_compare_matrix_parent[i] / sum_min_compare_matrix_parent

    return weighted_min_compare_matrix_parent


weighted_min_compare_matrix_parent = get_weighted_compare_matrix(np_criteria_parent_original_with_text,
                                                                 original_importance_parent_with_text)

weighted_min_compare_matrix_sub_problems = {}

# we have just finished doing parent calculations. No going on sub problems
for criteria in np_criteria_parent_original_with_text:
    criteria_name = criteria[0]

    np_criteria_sub_arr, np_criteria_sub_original_with_text, original_sub = my_math.read_matrix(
        "{}{}".format(current_problem_path, "criteria_{}.txt".format(criteria_name)))

    print_new_section()
    print("criteria \r\n{}".format(np_criteria_sub_original_with_text))

    np_importance_sub_arr, original_importance_sub_with_text, original_importance_sub = my_math.read_matrix(
        "{}{}".format(current_problem_path, "importance_{}.txt".format(criteria_name)))

    weighted_min_compare_matrix_sub_problems[criteria_name] = get_weighted_compare_matrix(np_criteria_sub_original_with_text,
                                                                                     original_importance_sub_with_text)

importance_array = my_math.generate_importance_correlation_matrix(original_importance)

np_criteria_arr = {}
np_criteria_original_with_text = {}
original = {}

for parent in np_criteria_original_with_text:
    np_criteria_arr[parent], np_criteria_original_with_text[parent], original[parent] = my_math.read_matrix(
        "{}{}".format(current_problem_path, "importance_{}.txt".format(parent)))

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
