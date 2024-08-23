"""
地震动记录类，采用GroundMotion包选波得到的实例属于该类，与生存的pickle波库配套使用
"""
from math import ceil
from typing import Literal
from PIL import Image
from pathlib import Path

import dill as pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['Records', 'records_splitting']


class Records:
    def __init__(self, name: str=None):
        self.name: str = name  # 小波库名
        self.N_gm: int = 0  # 地震动数量
        self.info: pd.DataFrame = None  # 地震动信息
        self.unscaled_data: list[np.ndarray] = []  # 未缩放时程序列
        self.unscaled_spec: list[np.ndarray] = []  # 未缩放反应谱数据
        self.SF: list[float] = []  # 缩放系数
        self.dt: list[float] = []  # 步长
        self.type_: list[Literal['A', 'V', 'D']] = []  # 数据类型(加速度, 速度, 位移)
        self.selecting_text: str = None
        self.target_spec: np.ndarray = None  # 目标谱(2列, 周期&加速度谱值)
        self.individual_spec: np.ndarray = None  # 各条波反应谱(1+N列, 周期&多列加速度谱值)(缩放后)
        self.mean_spec: np.ndarray = None  # 平均谱(2列, 周期&平均加速度谱值)(缩放后)
        self.img: Image = None  # 反应谱对比图
        self.spitted_times = 0  # 被切分的次数

    def _add_record(self,
            unscaled_data: np.ndarray,
            unscaled_spec: np.ndarray,
            SF: float,
            dt: float,
            type_: Literal['A', 'V', 'D']):
        """记录一条地震动

        Args:
            unscaled_data (np.ndarray): 无缩放的时程序列
            unscaled_spec (np.ndarray): 无缩放的反应谱
            SF (float): 缩放系数
            dt (float): 步长
            type_ (Literal['A', 'V', 'D']): 数据类型(加速度, 速度, 位移)
        """
        self.N_gm += 1
        self.unscaled_data.append(unscaled_data)
        self.unscaled_spec.append(unscaled_spec)
        self.SF.append(SF)
        self.dt.append(dt)
        self.type_.append(type_)

    def _to_pkl(self, file_name: str, folder: Path | str):
        """导出实例到pickle文件

        Args:
            file_name (str): 文件名(不带后缀)
            folder (Path | str): 文件夹路径
        """
        print(f'正在写入pickle文件(.records)...\r', end='')
        folder = Path(folder)
        if not folder.exists():
            raise FileExistsError(f'{str(folder.absolute())}不存在！')
        file = folder / f'{file_name}.records'
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print(f'已导出 {file.absolute().as_posix()}')

    def plot_spectra(self):
        """绘制反应谱曲线
        """
        T = self.individual_spec[:, 0]
        label = 'Individual'
        for col in range(1, self.individual_spec.shape[1]):
            Sa = self.individual_spec[:, col]
            plt.plot(T, Sa, color='#A6A6A6', label=label)
            if label:
                label = None
        plt.plot(self.target_spec[:, 0], self.target_spec[:, 1], label='Target', color='black', lw=3)
        plt.plot(self.mean_spec[:, 0], self.mean_spec[:, 1], color='red', label='Mean', lw=3)
        plt.xlim(min(self.target_spec[:, 0]), max(self.target_spec[:, 0]))
        plt.title('Selected records')
        plt.xlabel('T [s]')
        plt.ylabel('Sa [g]')
        plt.legend()
        plt.show()

    def get_unscaled_records(self) -> zip:
        """导出未缩放的时程

        Returns:
            zip[tuple[np.ndarray, float]]: 时程序列，步长

        Examples:
            >>> for data, dt in get_unscaled_records():
                    print(data.shape, dt)
        """
        return zip(self.unscaled_data, self.dt)

    def get_scaled_records(self) -> zip:
        """导出缩放后的时程

        Returns:
            zip[tuple[np.ndarray, float]]: 时程序列，步长

        Examples:
            >>> for data, dt in get_scaled_records():
                    print(data.shape, dt)
        """
        scaled_data = []
        for i, sf in enumerate(self.SF):
            scaled_data.append(self.unscaled_data[i] * sf)
        return zip(scaled_data, self.dt)
        
    def get_normalised_records(self) -> zip:
        """导出归一化的时程

        Returns:
            zip[tuple[np.ndarray, float]]: 时程序列，步长

        Examples:
            >>> for data, dt in get_normalised_records():
                    print(data.shape, dt)
        """
        normalised_data = []
        for i in range(self.N_gm):
            normalised_data.append(self._normalisation(self.unscaled_data[i]))
        return zip(normalised_data, self.dt)

    @staticmethod
    def _normalisation(data: np.ndarray):
        return data / np.max(np.abs(data))
    
    def show_info(self):
        """展示地震动信息"""
        print(f'Number of ground motions: {self.N_gm}\n')
        print(self.info)

    def get_record_name(self) -> list[str]:
        """获取地震动名称

        Returns:
            list[str]: 地震动名称
        """
        return self.info['earthquake_name'].to_list()
    
    
def records_splitting(num: int, records: Records=None, records_file: Path | str=None, to_file: bool | Path=False):
    """切分地震动数据库

    Args:
        num (int): 切分数量
        records (Records): 地震动记录数据库实例
        records_file (Path | str, optional): .records文件路径，默认None
        to_file (bool, optional): 切分后是否保存至文件，默认False，可指定保存文件夹路径
    """
    # sys.path.append(Path(__file__).parent.absolute().as_posix())
    if records is not None:
        file_name = records.name
    elif records_file is not None:
        records_file = Path(records_file)
        file_name = records_file.stem
        with open(records_file, 'rb') as f:
            records: Records = pickle.load(f)
    else:
        raise RecordError('必须指定参数`records`或`records_file`中的其中一个')
    if records.N_gm < num:
        raise RecordError('数据库中地震动数量小于切分数量')
    size = ceil(records.N_gm / num)  # 切分后每个数据库的地震动数量
    ls_new_records = []
    for idx in range(num):
        idx_start = idx * size
        idx_end = min(idx_start + size, records.N_gm)
        new_records = Records(records.name)
        new_records.N_gm = idx_end - idx_start
        new_records.info = records.info.iloc[idx_start: idx_end]
        new_records.unscaled_data = records.unscaled_data[idx_start: idx_end]
        new_records.unscaled_spec = records.unscaled_spec[idx_start: idx_end]
        new_records.SF = records.SF[idx_start: idx_end]
        new_records.dt = records.dt[idx_start: idx_end]
        new_records.type_ = records.type_[idx_start: idx_end]
        new_records.selecting_text = records.selecting_text
        new_records.target_spec = records.target_spec
        new_records.individual_spec = records.individual_spec[:, idx_start + 1: idx_end + 1]
        new_records.mean_spec = records.mean_spec
        new_records.mean_spec[:, 1] = np.mean(new_records.individual_spec[:, 1:], axis=1)
        new_records.img = records.img
        if not hasattr(records, 'spitted_times'):
            spitted_times_old = 0
        else:
            spitted_times_old = records.spitted_times
        new_records.spitted_times = spitted_times_old + 1
        ls_new_records.append(new_records)
        if to_file:
            new_records._to_pkl(f'{file_name}_{idx+1}', to_file)
    return tuple(ls_new_records)


class RecordError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'RecordError: {self.message}'

