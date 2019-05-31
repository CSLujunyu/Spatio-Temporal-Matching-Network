import numpy as np

def get_p_at_n_in_m(data, n, m, ind):
    curr = data[ind:ind+m]
    curr = sorted(curr, key = lambda x:x[0], reverse=True)
    flag = np.sum(np.array(curr)[:n,1])
    if flag == 1:
        return 1
    return 0

def MRR(data,ind):
    curr = data[ind:ind + 100]
    curr = sorted(curr, key=lambda x: x[0], reverse=True)
    for n, item in enumerate(curr):
        if item[1] == 1:
            return 1/(n+1)

def evaluate(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split("\t")
            if len(tokens) != 3:
                continue

            data.append([float(tokens[1]), int(tokens[2])])

    p_at_1_in_100 = 0.0
    p_at_2_in_100 = 0.0
    p_at_5_in_100 = 0.0
    p_at_10_in_100 = 0.0
    p_at_50_in_100 = 0.0
    mrr = 0.0

    length = int(len(data) / 100)
    for i in range(length):
        ind = i * 100

        p_at_1_in_100 += get_p_at_n_in_m(data, 1, 100, ind)
        p_at_2_in_100 += get_p_at_n_in_m(data, 2, 100, ind)
        p_at_5_in_100 += get_p_at_n_in_m(data, 5, 100, ind)
        p_at_10_in_100 += get_p_at_n_in_m(data, 10, 100, ind)
        p_at_50_in_100 += get_p_at_n_in_m(data, 50, 100, ind)
        mrr += MRR(data, ind)

    p_at_1_in_100 = p_at_1_in_100 / length
    p_at_2_in_100 = p_at_2_in_100 / length
    p_at_5_in_100 = p_at_5_in_100 / length
    p_at_10_in_100 = p_at_10_in_100 / length
    p_at_50_in_100 = p_at_50_in_100 / length
    mrr = mrr / length
    average = (p_at_10_in_100 + mrr) / 2

    return (p_at_1_in_100, p_at_2_in_100, p_at_5_in_100, p_at_10_in_100, p_at_50_in_100, mrr, average)
