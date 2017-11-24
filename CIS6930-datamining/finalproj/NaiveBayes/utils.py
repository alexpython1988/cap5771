'''functions used for calculation in naive bayes'''

def update_mean(u1, u2, n1, n2):
    return (u1 * n1 + u2 * n2) / (n1 + n2)

def update_variance(s1, s2, u1, u2, n1, n2):
    u = (u1 * n1 + u2 * n2) / (n1 + n2)
    return ((s1 + u1 ** 2) * n1 + (s2 + u2 ** 2) * n2) / (n1 + n2) - u ** 2

def evaluation(pred_label, true_label):
    n = len(pred_label)
    np = len(true_label)

    if n != np:
        raise ValueError("Predict and True label numbers must be same!")

    matched = 0
    for each in zip(pred_label, true_label):
        if each[0] == each[1]:
            matched += 1

    return "{:.3f}".format(1.0 * matched / n)
