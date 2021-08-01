import numpy as np
import matplotlib.pyplot as plt


# add ones column to x matrix
def add_ones(x):
    m = x.shape[0]
    ones = np.ones((m, 1))
    temp = np.append(ones, x, axis=1)
    return temp


def get_h_theta(x, theta):
    return x.dot(theta)


def get_j_theta(x, y, theta):
    m = x.shape[0]
    h_theta = get_h_theta(x, theta)
    j_theta = h_theta - y
    j_theta = np.square(j_theta)
    j_theta = np.sum(j_theta)
    return j_theta / (2 * m)


def get_grad_j_theta(x, y, theta):
    m = x.shape[0]
    xt = x.transpose()
    grad = x.dot(theta)
    grad = grad - y
    grad = xt.dot(grad)
    grad = grad / m
    return grad


def gradient_descent(x, y, a, iterations):
    x_ones = add_ones(x)
    n = x_ones.shape[1]
    theta = np.ones((n, 1))
    x_list = []
    y_list = []
    k = 0

    prev_j_theta = 0
    j_theta = get_j_theta(x_ones, y, theta)
    j_theta_trash = 0.01
    j_theta_diff = abs(j_theta - prev_j_theta)

    while k < iterations and j_theta_diff > j_theta_trash:
        gradient = get_grad_j_theta(x_ones, y, theta)
        if k != 0:
            prev_j_theta = j_theta
        j_theta = get_j_theta(x_ones, y, theta)
        j_theta_diff = abs(j_theta - prev_j_theta)
        print(j_theta)

        theta = theta - a * gradient
        x_list.append(k)
        y_list.append(j_theta)
        k += 1

    return x_list, y_list


def GD(x, y, iterations):
    x1, y1 = gradient_descent(x, y, 0.1, iterations)
    x2, y2 = gradient_descent(x, y, 0.01, iterations)
    x3, y3 = gradient_descent(x, y, 0.001, iterations)
    plt.plot(x1, y1, 'r', label='a = 0.1')
    plt.plot(x2, y2, 'g', label='a = 0.01')
    plt.plot(x3, y3, 'b', label='a = 0.001')
    plt.xlabel('iterations')
    plt.ylabel('J theta')
    plt.legend()
    plt.show()


def stochastic(x, y, a, iterations, epsilon=0.001):
    m = x.shape[0]
    ones = np.ones((m, 1))
    xOnes = np.append(ones, x, axis=1)

    n = xOnes.shape[1]
    prev_theta = np.zeros((n, 1))
    theta = np.ones((n, 1))
    theta_dif = np.linalg.norm(theta - prev_theta)  # norma

    j = get_j_theta(xOnes, y, theta)
    prev_j = 0
    k = 0
    x_list = []  # iterations
    y_list = []  # J
    i = 0
    while ((k < iterations) and (theta_dif > epsilon) and (abs(j - prev_j) > epsilon)):
        i = i % m
        if (k != 0):
            prev_theta = theta
            prev_j = j
        x_list.append(k)
        j = get_j_theta(xOnes, y, theta)
        y_list.append(j)
        h = get_h_theta(xOnes[i], theta)
        h = h - y[i]

        xt = a * xOnes[i] * h
        xt = xt.reshape((n, 1))
        theta = theta - xt

        theta_dif = np.linalg.norm(theta - prev_theta)  # norma
        k += 1
        i += 1
    return x_list, y_list


def ST(x, y):
    x1, y1 = stochastic(x, y, 0.01, 500)
    plt.plot(x1, y1, 'r', label='stochastic')
    plt.xlabel('iterations')
    plt.ylabel('J theta')
    plt.legend()
    plt.show()


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    newY = np.reshape(y, (X.shape[0], 1))
    print(X.shape)
    print(newY.shape)
    data = np.append(X, newY, axis=1)
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


def mini_batch(x, y, learning_rate=0.1, batch_size=32):
    x_ones = add_ones(x)
    n = x_ones.shape[1]
    theta = np.ones((n, 1))
    xt = x_ones.transpose()

    x_list = []
    y_list = []
    k = 0
    max_iters = 3
    for itr in range(max_iters):
        mini_batches = create_mini_batches(x_ones, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta = theta - learning_rate * get_grad_j_theta(X_mini, y_mini, theta)
            j_theta = get_j_theta(x_ones, y, theta)
            x_list.append(k)
            y_list.append(j_theta)
            k += 1
    return x_list, y_list


def MB(x, y):
    x1, y1 = mini_batch(x, y)
    plt.plot(x1, y1, 'r', label='mini batch')
    plt.xlabel('iterations')
    plt.ylabel('J theta')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get data
    x = np.genfromtxt('.//data.csv', delimiter=',')
    y = x[:, -1]  # get last column
    x = np.delete(x, -1, 1)  # remove last column

    # get avr and std
    avr = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # normalize the data
    b = x - avr
    x = b / std

    # try
    # GD(x, y, 100)
    ST(x, y)
    # MB(x, y)
