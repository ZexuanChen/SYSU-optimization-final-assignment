import numpy as np
import matplotlib.pyplot as plt
import time
from math import *

def f0(A, b, x, lamda):
    return np.sum([np.linalg.norm(A[i].dot(x)-b[i]) for i in range(A.shape[0])], axis=0)/2 + lamda*np.linalg.norm(x, ord=1)
    
def soft_thresholding(x, threshold):
    '''软门限法算邻近点投影，x是邻近点'''
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def proximal_gradient_method(A, b, lamda, alpha=0.0001, max_iter=5000, tol=1e-5, if_draw=True):
    '''邻近点梯度法求解'''
    start_time = time.time()
    
    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解

    iterates = []  # 记录每步的解
    
    # 迭代求解
    for _ in range(max_iter):
        x_old = x.copy()

        gradient = np.zeros(d2)
        # 算梯度
        for i in range(n):
            gradient += A[i].T.dot(A[i].dot(x) - b[i])

        # 软门限法算argmin
        x = soft_thresholding(x - alpha * gradient, lamda * alpha)
        
        iterates.append(x)  # 记录解
        
        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break
    
    end_time = time.time()
    diff_time = end_time - start_time
    
    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]
    
    if if_draw:
        # 绘制距离图
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('proximal gradient method')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()
    
    print(f'proximal gradient using time(alpha={alpha}, lambda={lamda}): {diff_time}')  # 打印时间
    print(f'distance of proximal gradient x_opt and x_true(alpha={alpha}, lambda={lamda}): ' 
          f'{np.linalg.norm(x - x_true)}', end='\n\n')  # 打印二范数误差
    return x, distances_to_true, distances_to_opt

def admm(A, b, lamda, C=1, max_iter=1000, tol=1e-5, if_draw=True):
    '''交替方向乘子法求解'''
    start_time = time.time()
    
    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解
    y = np.zeros(d2)
    v = np.zeros(d2)
    
    iterates = []  # 记录每步的解
    r = []
    
    for _ in range(max_iter):
        x_old = x.copy()

        # 更新x
        x = np.linalg.inv(np.sum([A[i].T.dot(A[i]) for i in range(n)], axis=0) + C*np.eye((d2)))
        x = x.dot(np.sum([A[i].T.dot(b[i]) for i in range(n)], axis=0) + C*y - v)
        # 更新y
        y = soft_thresholding(x + v/C, lamda/C)
        # 更新v
        v += C*(x-y)

        iterates.append(x)
        r.append(f0(A, b, x, lamda))
        
        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break
    
    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]
    
    end_time = time.time()
    diff_time = end_time - start_time
    
    if if_draw:
        # 绘制距离变化图
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('alternating direction method of multipliers')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()
    
    # # 绘制函数值变化图
    # plt.plot(r)
    # plt.title('alternating direction method of multipliers')
    # plt.xlabel('iteration')
    # plt.ylabel('f0(x)')
    # plt.grid()
    # plt.legend()
    # plt.show()
    
    print(f'admm using time(C={C}, lambda={lamda}): {diff_time}')  # 打印所用时间
    print(f'distance of admm x_opt and x_true(C={C}, lambda={lamda}): ' 
          f'{np.linalg.norm(x - x_true)}', end='\n\n')  # 打印二范数误差
    return x, distances_to_true, distances_to_opt
    
def subgradient(A, b, lamda, alpha=0.0001, max_iter=5000, tol=1e-5, if_draw=True):
    '''次梯度法求解'''
    start_time = time.time()
    
    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解
    
    iterates = []  # 记录每步的解
    
    for _ in range(max_iter):
        x_old = x.copy()

        # 次梯度
        g = np.empty_like(x)
        for i, data in enumerate(x):
            if data == 0:
                g[i] = 2 * np.random.random() - 1  # [-1, 1]
            else:
                g[i] = np.sign(x[i])
        g *= lamda
        g += np.sum([A[i].T.dot(A[i].dot(x) - b[i]) for i in range(n)], axis=0)
        # 更新x
        x = x - alpha*g
        
        iterates.append(x)
        
        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break
    
    end_time = time.time()
    diff_time = end_time - start_time
    
    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]
    
    if if_draw:
        # 绘制距离变化图
        plt.figure()
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('subgradient')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()
    
    print(f'subgradient using time(alpha={alpha}, lambda={lamda}): {diff_time}')  # 打印时间
    print(f'distance of subgradient x_opt and x_true(alpha={alpha}, lambda={lamda}): ' 
          f'{np.linalg.norm(x - x_true)}', end='\n\n')  # 打印二范数误差
    return x, distances_to_true, distances_to_opt

def adjust_lamda(A, b, lamdas, method):
    '''
    调整正则化参数，lamdas是参数列表，method决定用哪个优化算法，同时作为绘制图形的suptitle，
    method只能取值'proximal gradient', 'admm' 或'subgradient'
    '''
    fig, axes = plt.subplots(int(sqrt(len(lamdas))), ceil(len(lamdas)/2), figsize=(12, 8))  # 创建多个子图
    # 画每一个参数值对应的子图
    for i, lamda in enumerate(lamdas):
        if method == 'proximal gradient': r1 = proximal_gradient_method(A, b, lamda, if_draw=False)
        elif method == 'admm': r1 = admm(A, b, lamda, if_draw=False)
        elif method == 'subgradient': r1 = subgradient(A, b, lamda, if_draw=False)
        
        row, col = divmod(i, ceil(len(lamdas)/2))  # 计算子图位置
        axes[row, col].plot(r1[1], label='distance to true')
        axes[row, col].plot(r1[2], label='distance to opt')
        axes[row, col].set_title(r"$\lambda = $" + f"{lamda}")
        axes[row, col].set_xlabel('iteration')
        axes[row, col].set_ylabel('distance')
        axes[row, col].grid()
        axes[row, col].legend()
        
    plt.suptitle(method)
    plt.tight_layout()
    plt.show()


np.random.seed(0)
num = 10 
d1 = 5 
d2 = 200
lamda = 0.01  # 可以调整

# 随机生成矩阵A、x的真值、和向量b
A = np.array([np.random.normal(0, 1, (d1, d2)) for _ in range(num)])
x_true = np.zeros(d2)
# 随机选择5个位置非0，其他位置为0
nonzero_indices = np.random.choice(d2, 5, replace=False)
x_true[nonzero_indices] = np.random.normal(0, 1, d1)
b = np.array([A[i].dot(x_true) + np.random.normal(0, 0.1, d1) for i in range(num)])

# 三种算法求解
x_opt1 = proximal_gradient_method(A, b, lamda)[0]
x_opt2 = admm(A, b, lamda, if_draw=False)[0]
x_opt3 = subgradient(A, b, lamda)[0]

# 调整正则化参数
# lamdas = [0.001, 0.01, 0.1, 1, 10, 100]
lamdas = [0.01, 0.1, 1, 5, 50, 100]
adjust_lamda(A, b, lamdas, 'admm')