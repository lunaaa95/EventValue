import pandas as pd
import numpy as np
import math
from dataclasses import dataclass

from .. import Hawkes as hk
from .ggs import GGS


@dataclass
class Storyline:
    """

    time:
        事件的时间轴
        shape: (T,)
        
    strength:
        事件的强度
        shape: (T,)
    """
    news: pd.Series
    time: pd.Series|pd.DatetimeIndex
    related: pd.Series
    topic: str|list[str] = None
    embedding: np.ndarray = None
    strength: np.ndarray = None


class StrengthAnalyzer:
    def __init__(self, ) -> None:
        pass
    
    def sort_by_time(self, storyline: Storyline):
        df = pd.DataFrame({
            'news': storyline.news,
            'time': storyline.time,
            'related': storyline.related
        }).sort_values(by='time')
        
        storyline.news = df['news']
        storyline.time = df['time']
        storyline.related = df['related']

        return storyline
    
    def estimate(self, storyline: Storyline):
        """通过新闻时间估计故事线的强度变化
        建模为Hawkes点过程
        """

        # 拷贝时间戳序列,将重复时间戳间隔1秒
        ts = storyline.time.copy()
        delta = pd.Timedelta(seconds=1)
        for i, t in enumerate(ts):
            if i > 0:
                increment = t - ts[i-1]
                if increment <= delta:
                    ts[i] = ts[i-1] + delta

        # 时间戳转换为浮点数,缩放单位为千秒
        ts = ts.to_numpy().astype(float)
        ts /= 1e12

        # 设置首个时间戳前1天为原点
        start = ts[0] - 86.4
        ts -= start
        # 设置最后时间戳后1天为终点
        end = math.ceil(ts[-1] + 86.4)

        # 估计Hawkes过程参数
        model = hk.estimator()
        model.set_kernel('exp')
        model.set_baseline('const')
        model.fit(ts, itv=[0, end])

        # 推算每个事件的强度
        _, lkb, _ = model.tl()
        strength = lkb[::30]  # Hawkes模块两次事件采样间隔30个格子,但不同的两组事件间采样格子长度不相等

        storyline.strength = strength

        return storyline


class Breakpoint:
    @staticmethod
    def find_breakpoints(storyline: Storyline, Kmax=10, lamb=1e-4, beta=1e-3, return_index=False):
        """寻找可能的事件强度转折点
        storyline:
            故事线
        
        Kmax:
            建议的转折点数目上限
            type: int
        
        lamb:
            转折点检测超参
        
        beta:
            转折点数量超参,越小转折点越少
        """

        # 拷贝时间戳序列,时间戳转换为浮点数,缩放单位为千秒
        ts = storyline.time.to_numpy().astype(float) / 1e12

        # 尝试划分转折点
        all_bps, lls = GGS(storyline.strength.reshape(1, -1), Kmax=Kmax, lamb=lamb)

        # 选择转折点
        bps_index = Breakpoint.select_breakpoint(ts, all_bps, lls, beta)[1:-1]

        if return_index:
            return bps_index
        else:
            return storyline.time[bps_index]
    
    @staticmethod
    def select_breakpoint(time_index, all_breakpoints, likelihoods, beta):
        likelihoods = np.array(likelihoods)
        # 计算beta使得time delta loss能和lls在相近数量级
        gamma = np.mean(likelihoods[1:] - likelihoods[:-1])
        total_lls = []
        for bps, ll in zip(all_breakpoints, likelihoods):
            tdl = Breakpoint.time_delta_loss(time_index, bps, beta)
            total_lls.append( -tdl * gamma + ll)
        best_i = np.argmax(total_lls)
        return all_breakpoints[best_i]
    
    @staticmethod
    def time_delta_loss(time_index, breakpoints, beta):
        # 将分割点还原到时间轴上
        time_breakpoints = np.empty(len(breakpoints))
        time_breakpoints[:-1] = time_index[breakpoints[:-1]]
        time_breakpoints[-1] = time_index[-1] + 1e-3

        # 计算两个分割点之间的时差
        time_delta = time_breakpoints[1:] - time_breakpoints[:-1]

        # 损失函数
        loss = np.sum(np.exp(-beta * time_delta))

        return loss
