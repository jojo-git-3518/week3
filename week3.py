
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

titanic_f = 'titanic/train.csv'
content = pd.read_csv(titanic_f)

content = content.dropna()
age_with_fares = content[(content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)]
age = age_with_fares['Age']
fare = age_with_fares['Fare']
sub_ages = age_with_fares['Age'].tolist()
sub_fares = age_with_fares['Fare'].tolist()
#plt.scatter(sub_ages, sub_ages)
#plt.show()


def func(age_new, k, b):
    return k * age_new + b


def loss(y, yhat):

    """
    :param y:  real fares
    :param yhat: 估计的
    :return: 二者之间的差距
    """
    return np.mean(np.abs(y -yhat))


def derivate_k(y, y_hat, x ):
    abs_value = [1 if (y_i > yhat_i) else -1 for y_i, yhat_i in zip(y, y_hat) ]
    return np.mean([ a * -x_i for a, x_i in zip(abs_value, x)])
def derivate_b(y, y_hat):
    abs_value = [1 if (y_i > yhat_i) else -1 for y_i, yhat_i in zip(y, y_hat) ]
    return np.mean([ a * -1 for a in abs_value])


loop_times=10000
min_error = float('inf')   # 正无穷

change_directions=[
    (+1, -1),
    (+1, +1),
    (-1, -1),
    (-1, +1)
]
best_direction = None
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10


def step(): return random.random() *1


k_direction, b_direction= random.choice(change_directions)
losses = []
learning_rate=1e-2

while loop_times > 0:
    # k_step = k_direction * step()
    #b_step = b_direction * step()
    #k_hat = k_hat + k_step
    #b_hat = b_hat + b_step
    k_hat = k_hat - learning_rate * derivate_k(sub_fares, func(age, k_hat, b_hat), sub_ages)
    b_hat =b_hat - learning_rate * derivate_b(sub_fares, func(age, k_hat, b_hat))

    # k_hat = random.random() * 20 - 10
    # b_hat = random.random() * 20 - 10

    # k_hat = random.randint(-10, 10)
    # b_hat = random.randint(-10, 10)
    estimate_fares = func(age, k_hat, b_hat)
    performance = loss(fare, estimate_fares)
    if performance < min_error:
        best_k, best_b = k_hat, b_hat
        min_error = performance
        losses.append(min_error)

    #else:
        #k_direction, b_direction = random.choice(list(set(change_directions) - {(k_direction, b_direction)}))
    loop_times = loop_times - 1
    print('loop ={}'.format(10000 - loop_times))
    print('f(age) ={} * age + {},with error rate :{}'.format(best_k, best_b, min_error))





#plt.scatter(sub_ages, sub_ages)
plt.plot(range(len(losses)), losses)
#plt.plot(sub_ages, estimate_fares)
plt.show()



