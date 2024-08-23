"""
计算反应谱
（采用Nigam-Jennings精确解）
"""
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['spectrum', 'scale_record']


def spectrum(
        ag: np.ndarray,
        dt: float,
        T: np.ndarray=None,
        zeta: float=0.05
    ):
    """计算给定地震动的反应谱

    Args:
        ag (np.ndarray): 加速度时程
        dt (float): 步长
        T (np.ndarray, optional): 周期序列，若不指定则默认为0-6s
        zeta (float, optional): 阻尼比，默认0.05

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 加速度谱，速度谱，位移谱
    """
    if T is None:
        T = np.arange(0, 6.01, 0.01)
    if T[0] == 0:
        T = T[1:]
        mark = 1
    else:
        mark = 0
    N = len(T)
    w = 2 * np.pi / T
    wd = w * np.sqrt(1 - zeta**2)
    n = len(ag)
    u = np.zeros((N, n))  # N x n
    v = np.zeros((N, n))  # N x n
    B1 = np.exp(-zeta * w * dt) * np.cos(wd * dt)  # N
    B2 = np.exp(-zeta * w * dt) * np.sin(wd * dt)  # N
    w_2 = 1.0 / w ** 2  # N
    w_3 = 1.0 / w ** 3  # N
    for i in range(n - 1):
        u_i = u[:, i]  # N
        v_i = v[:, i]  # N
        p_i = -ag[i]
        alpha_i = (-ag[i + 1] + ag[i]) / dt
        A0 = p_i * w_2 - 2.0 * zeta * alpha_i * w_3  # N
        A1 = alpha_i * w_2  # N
        A2 = u_i - A0  # N
        A3 = (v_i + zeta * w * A2 - A1) / wd  # N
        u[:, i+1] = A0 + A1 * dt + A2 * B1 + A3 * B2  # N
        v[:, i+1] = A1 + (wd * A3 - zeta * w * A2) * B1 - (wd * A2 + zeta * w * A3) * B2  # N  
    w_tile = np.tile(w, (n, 1)).T  # N x n
    a = -2 * zeta * w_tile * v - w_tile * w_tile * u  # N x n
    Sa = np.amax(abs(a), axis=1)
    Sv = np.amax(abs(v), axis=1)
    Sd = np.amax(abs(u), axis=1)
    if mark == 1:
        Sa = np.insert(Sa, 0, max(abs(ag)))
    return Sa, Sv, Sd

def scale_record(
        T_target: np.ndarray,
        Sa_target: np.ndarray,
        ag: np.ndarray,
        dt: float,
        scale_method: Literal['a', 'b', 'c', 'd', 'e'],
        scale_paras: float | tuple=None,
        zeta: float=0.05
    ):
    """对地震动加速度时程进行缩放以匹配反应谱
    `scale_method`:
    * [a] - 不缩放
    * [b] - 按Sa(T0)缩放，scale_paras=T0
    * [c] - 按PGA缩放，scale_paras=PGA|None
    * [d] - 按周期范围Ta-Tb缩放，scale_paras=(Ta, Tb)
    * [e] - 指定缩放系数，scale_paras=SF

    Args:
        T_target (np.ndarray): 目标谱的周期序列
        Sa_target (np.ndarray): 目标谱的加速度谱值序列
        ag (np.ndarray): 加速度时程
        dt (float): 步长
        scale_method (Literal['a', 'b', 'c', 'd', 'e']): 缩放方法
        scale_paras (float | list, optional): 可选的缩放参数
        zeta (float, optional): 反应谱计算阻尼比，默认0.05

    Returns:
        float: 缩放系数
    """
    if not len(T_target) == len(Sa_target):
        raise ValueError('Error 1')
    Sa_record, _, _ = spectrum(ag, dt, T_target, zeta)
    if scale_method == 'a':
        # 不缩放
        SF = 1
    elif scale_method == 'b':
        # 按T0缩放
        T0 = scale_paras
        SF = _find_Sa(T_target, Sa_target, T0)[0] / _find_Sa(T_target, Sa_record, T0)[0]
    elif scale_method == 'c':
        # 按PGA缩放
        if scale_paras is None:
            PGA = Sa_target[0]
        else:
            PGA = scale_paras
        SF = PGA / Sa_record[0]
    elif scale_method == 'd':
        # 按周期范围Ta-Tb缩放
        Ta, Tb = scale_paras
        idx_a, idx_b = _find_Sa(T_target, Sa_record, Ta)[1], _find_Sa(T_target, Sa_record, Tb)[1]
        init_SF = 1.0  # 初始缩放系数
        learning_rate = 0.01  # 学习率
        num_iterations = 40000  # 迭代次数
        SF = _gradient_descent(Sa_record[idx_a: idx_b], Sa_target[idx_a: idx_b], init_SF, learning_rate, num_iterations)
    elif scale_method == 'e':
        # 指定缩放系数
        SF = scale_paras
    else:
        raise ValueError('`scale_method`参数错误！')
    return SF  # scale factor

def _find_Sa(T: np.ndarray, Sa: np.ndarray, T0: float):
    for i in range(len(T) - 1):
        if T[i] <= T0 <= T[i+1]:
            k = (Sa[i+1] - Sa[i]) / (T[i+1] - T[i])
            Sa0 = k * (T0 - T[i]) + Sa[i]
            idx = i
            return Sa0, idx
    else:
        raise ValueError('Sa not found!')

def _gradient_descent(a, b, init_SF, learning_rate, num_iterations):
    """梯度下降"""
    f = init_SF
    for _ in range(num_iterations):
        error = a * f - b
        gradient = 2 * np.dot(error, a) / len(a)
        f -= learning_rate * gradient
    return f


if __name__ == "__main__":

    ag = np.loadtxt('data/a.txt')
    data = np.loadtxt('data/RCF6S_DBE.txt')
    T_target = data[:, 0]
    Sa_target = data[:, 1]
    SF = scale_record(T_target, Sa_target, ag, 0.01, 'c')
    Sa, Sv, Sd = spectrum(ag, 0.01, T_target)
    # print(SF)
    plt.plot(T_target, Sa * SF)
    plt.plot(T_target, Sa_target)
    plt.show()

    
