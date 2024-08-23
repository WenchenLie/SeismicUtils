"""
对离散数组进行数值微分或积分
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid


__all__ = ['integrate', 'differential']


def integrate(
        y: list | np.ndarray,
        dx: float,
        n: int=1,
        baseline_correction: bool=False
    ) -> np.ndarray:
    """利用梯形公式计算等步长的数值积分

    Args:
        y (list | np.ndarray): 函数值序列
        dx (float): 步长
        n (int, optional): 积分次数，默认1
        baseline_correction (bool, optional): 是否进行基线修正，默认False

    Returns:
        np.ndarray: 原函数
    """
    for _ in range(n):
        y = cumulative_trapezoid(y, dx=dx, initial=0)
    if baseline_correction:
        baseline = np.linspace(0, y[-1], len(y))
        y -= baseline
    return y


def differential(
        y: list | np.ndarray,
        dx: float,
        n: int=1
    ) -> np.ndarray:
    """对数组进行数值微分

    Args:
        y (list | np.ndarray): 函数值序列
        dx (float): 步长
        n (int): 微分次数，默认1

    Returns:
        np.ndarray: 导数序列
    """
    for _ in range(n):
        dy_dx = (y[2:] - y[: -2]) / (2 * dx)
        dy_start = (y[1] - y[0]) / dx
        dy_end = (y[-1] - y[-2]) / dx
        y = np.concatenate((np.array([dy_start]), dy_dx, np.array([dy_end])))
    return y


