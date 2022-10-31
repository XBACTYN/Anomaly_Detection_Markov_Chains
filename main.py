import numpy as np
import pandas as pd
import sys


def get_data_from_TXT(data_path):
    # f=pd.read_csv(data_path,sep =":",header=None)
    f = pd.read_csv(data_path, sep=':', index_col=0, names=["User", "Data"])
    return f


def get_states(data):
    states = []
    for i, row in data.iterrows():
        states += row.Data.split(";")
    states_set = set(states)  # Убираем дубликаты состояний
    states_list = list(states_set)
    states_list.sort()
    return states_list  # Преобразуем в список для возможности индексирования


def state_probs_matrix(data, states):
    matrix = np.full((len(states), len(states)), 0.00001)

    for i in range(len(data) - 1):
        matrix[states.index(data[i]), states.index(
            data[i + 1])] += 1  # Считаем количество переходов между состояниями и заносим в таблицу

    for i in range(len(states)):  # Получаем вероятности.
        matrix[i] /= matrix[i].sum()

    # print(matrix)
    return matrix


def find_interval(data, matrix, window, states):
    probs = row_prob(data, matrix, window, states)
    interval = (np.min(probs), np.max(probs))
    return interval


def row_prob(data, matrix, window, states):
    probs = []  # Массив вероятностей оконных последовательностей. Первый элемент -вероятности первых N(10) чисел(с 0 до 9),второй элемент с 1 по 10 и тд.
    # Идем окном и высчитываем для каждого окна вероятность появления последовательности,расположенной в окне

    if len(data) > window:
        for i in range(len(data) - window):
            win_data = data[i: i + window]  # Срез с i по i+window. Само окно создано
            probs.append(window_prob(win_data, matrix, states))  # Считаем вероятность последовательности в окне
    else:
        probs.append(window_prob(data, matrix, states))  # если строка меньше чем окно

    return probs


def window_prob(data, matrix, states):
    prob = 1
    for i in range(len(data) - 1):
        prob *= matrix[states.index(data[i]), states.index(data[i + 1])]
    return prob


def find_anomaly(data_test, matrix, window, interval, states):
    probs = row_prob(data_test, matrix, window, states)

    for prob in probs:
        if not (interval[0] <= prob <= interval[1]):
            return 1
    return 0


def data_to_dict(data, data_true, data_fake):
    data_dict = {}
    true_dict = {}
    fake_dict = {}
    for i, row in data.iterrows():
        data_dict.update({row.name: row.Data.split(";")})
    for i, row in data_true.iterrows():
        true_dict.update({row.name: row.Data.split(";")})
    for i, row in data_fake.iterrows():
        fake_dict.update({row.name: row.Data.split(";")})

    return data_dict, true_dict, fake_dict


def process(win, data_dict, true_dict, fake_dict, states_list):
    window = win
    fake_result = []
    true_result = []
    prob_dict = {}

    for key, value in data_dict.items():
        data_row = value
        matrix = state_probs_matrix(data_row, states_list)
        prob_dict.update({key: matrix})
        interval = find_interval(data_row, matrix, window, states_list)
        if true_dict.get(key, False):
            data_true_row = true_dict.get(key)
            true_result.append(find_anomaly(data_true_row, matrix, window, interval, states_list))
        else:
            true_result.append(404)

        if fake_dict.get(key, False):
            data_fake_row = fake_dict.get(key)
            fake_result.append(find_anomaly(data_fake_row, matrix, window, interval, states_list))
        else:
            fake_result.append(404)

    t_errors = np.count_nonzero(true_result)
    f_errors = np.count_nonzero(fake_result)

    return true_result, fake_result, t_errors, f_errors,prob_dict


def print_matrix(key,data_dict,prob_dict,states):
    vect = data_dict.get(key)
    matrix = prob_dict.get(key)
    print('user ',key)
    print('states ',states)
    print('vect',vect)
    pr_matrix = np.c_[list(map(int,states)),matrix]
    pr_states = np.r_[[0],np.asarray(list(map(int,states)))]
    print( pr_states)
    pr_matrix = np.r_[[list(map(int,pr_states))],pr_matrix]
    for i in range(len(pr_matrix)):
        for j in range(len(pr_matrix[i])):
            print("{:4f}\t".format(pr_matrix[i][j]), end="")
        print()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    window = 5
    data = get_data_from_TXT("C:/Users/user/Downloads/data.txt")
    data_true = get_data_from_TXT("C:/Users/user/Downloads/data_true.txt")
    data_fake = get_data_from_TXT("C:/Users/user/Downloads/data_fake.txt")

    data = data.sort_values(by=["User"])
    data_true = data_true.sort_values(by=["User"])
    data_fake = data_fake.sort_values(by=["User"])

    print("data ", data)
    print("data true ", data_true)
    print("data fake", data_fake)

    states_list = get_states(data)

    print("Окно: ",window)
    print("Всего ", len(states_list), "неповторяющихся значений:", states_list)

    data_dict, true_dict, fake_dict = data_to_dict(data, data_true, data_fake)

    true_result, fake_result, t_errors, f_errors,prob_dict = process(window, data_dict, true_dict, fake_dict, states_list)

    print_matrix('user15',data_dict,prob_dict,states_list)

    print("true\t", true_result, "\nfake\t", fake_result)
    print("true_data errors: ", t_errors, "\nfake_data errors: ", f_errors)
