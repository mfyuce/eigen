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
        if isinstance(importance_parent[importance_index_b][importance_index_a], int):
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
parent_options_size = len(original_options_with_text)
options_sizes = {"parent": parent_options_size}
options_importance = {}
weighted_sum_parent_options = np.zeros(parent_options_size)

# we have just finished doing parent calculations. No going on sub problems
for criteria in np_criteria_parent_original_with_text:
    criteria_name = criteria[0]
    importance_index_parent_options = find_criteria_index(criteria_name, np_criteria_parent_original_with_text)
    parent_weight_value  =weighted_min_compare_matrix_parent[importance_index_parent_options]
    np_criteria_sub_arr, np_criteria_sub_original_with_text, original_sub = my_math.read_matrix(
        "{}{}".format(current_problem_path, "criteria_{}.txt".format(criteria_name)))
    sub_options_size = len(np_criteria_sub_original_with_text)
    options_sizes[criteria_name] = sub_options_size
    print_new_section()
    print("criteria \r\n{}".format(np_criteria_sub_original_with_text))

    np_importance_sub_arr, original_importance_sub_with_text, original_importance_sub = my_math.read_matrix(
        "{}{}".format(current_problem_path, "importance_{}.txt".format(criteria_name)))

    weighted_min_compare_matrix_sub_problems[criteria_name] = get_weighted_compare_matrix(
        np_criteria_sub_original_with_text,
        original_importance_sub_with_text)

    np_option_importance_sub_arr, original_option_importance_sub_with_text, original_option_importance_sub = my_math.read_matrix(
        "{}{}".format(current_problem_path, "options_importance_{}.txt".format(criteria_name)))
    options_size_tupple = (sub_options_size, parent_options_size)
    options_importance[criteria_name] = np.zeros(options_size_tupple)
    for option_importance in original_option_importance_sub_with_text:
        importance_value = my_math.covert_to_float(option_importance[0])
        sub_option_name = option_importance[1]
        parent_option_name = option_importance[2]
        importance_index_parent_options = find_criteria_index(parent_option_name, original_options_with_text)
        importance_index_sub_options = find_criteria_index(sub_option_name, np_criteria_sub_original_with_text)
        options_importance[criteria_name][importance_index_sub_options][importance_index_parent_options] = importance_value

    weighted_parent_options = np.zeros(options_size_tupple)
    for i in range(sub_options_size):
        sub_waighted_value = weighted_min_compare_matrix_sub_problems[criteria_name][i]
        for j in range(parent_options_size):
            weighted_parent_options[i][j] = parent_weight_value * options_importance[criteria_name][i][j] * sub_waighted_value

    weighted_sum_parent_options = np.add(weighted_sum_parent_options, np.sum(weighted_parent_options, axis=0))


print_new_section()
print("Unordered Solution Matrix \r\n{}".format(weighted_sum_parent_options))
solution = {}
for i in range(parent_options_size):
    option_name = original_options_with_text[i][0]
    solution[option_name] = weighted_sum_parent_options[i]

print("Solution Order \r\n{}".format({k: v for k, v in sorted(solution.items(), key=lambda item: -item[1])}))
