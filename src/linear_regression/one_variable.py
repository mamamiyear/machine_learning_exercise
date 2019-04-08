# -*- coding: utf-8 -*-
__author__ = "mamamiyear"

import matplotlib.pyplot as plt
from linear_regression.one_vaiable_dataset_creator import \
    create_feature, create_target, min_uniform
from random import randint


def train(feature_array, target_array, train_speed=0.3, train_times=1000):
    if len(feature_array) != len(target_array):
        raise Exception("train error: feature array length not equal target array length.")

    theta0 = randint(0, 10) * 1.0
    theta1 = randint(0, 10) * 1.0

    length = len(feature_array)

    plt.ion()
    plt.xlim(min(feature_array), max(feature_array))
    plt.ylim(min(target_array), max(target_array))
    plt.scatter(feature_array, target_array)
    plt.pause(1)

    line = None

    for i in range(train_times):
        gradient_theta0 = 0.0
        gradient_theta1 = 0.0
        for x, y in zip(feature_array, target_array):
            gradient_theta0 += theta0 + theta1 * x - y
            gradient_theta1 += (theta1 * x + theta0 - y) * x
        gradient_theta0 /= length
        gradient_theta1 /= length
        theta0 -= train_speed * gradient_theta0
        theta1 -= train_speed * gradient_theta1

        # 画图
        if i % 20 == 0:
            print("train times %4d: (%9.6f, %9.6f)" % (i, theta0, theta1))
            if line:
                line.remove()
                del line
            _array = theta0 + theta1 * feature_array
            line, = plt.plot(feature_array, _array)
            plt.pause(0.1)

        # 检查是否平均误差小于阈值
    return theta0, theta1


def average_deviation(array1, array2):
    if len(array1) != len(array2):
        raise Exception("loss_function error: array1 length not equal array2 length.")
    deviation = 0
    length = len(array2)
    for i in range(length):
        deviation += (array1[i] - array2[i]) ** 2
    deviation /= (2 * length)
    return deviation


if __name__ == "__main__":
    # 定义
    theta0 = 6
    theta1 = -3

    trainset_number = 100
    testset_number = 20

    feature_noise = 1
    target_noise = 0.1

    # 产生训练集
    feature_array = create_feature(0, 100, trainset_number, feature_noise)
    feature_array = min_uniform(feature_array)  # 归一化
    print("--------------------------------------------")
    target_array = create_target(theta0, theta1, feature_array, target_noise)
    print("--------------------------------------------")
    # 产生测试集
    test_feature_array = create_feature(100, 120, testset_number)
    test_target_array = create_target(theta0, theta1, test_feature_array)

    # 训练模型
    trained_theta0, trained_theta1 = train(feature_array, target_array)
    trained_target_array = trained_theta0 + trained_theta1 * test_feature_array

    # 对比效果
    deviation = average_deviation(test_target_array, trained_target_array)
    print("average deviation: %f" % deviation)

    plt.ioff()
    plt.show()
    pass
