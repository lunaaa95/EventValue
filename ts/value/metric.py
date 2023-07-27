import numpy as np
from dataclasses import dataclass

from .ggs import GGS
from .model import Result


@dataclass
class ValueTimeRange:
    """记录偏离价值范围的时间段

    3种情况
    1. 先偏离后回归:
        deviation_time和return_time都非None
    2. 偏离后未回归:
        仅deviation_time非None
    3. 未偏离:
        deviation_time和return_time都为None
    """
    deviation_time: int|str = None
    return_time: int|str = None
    end_time: int|str = None
    deviation_ratio: float = None
    return_ratio: float = None


def find_value_return(result: Result, n_train: int, ratio_threshold=0.33):
    """寻找第一个回到估值范围内的子区间

    Result:
        估值结果
        type: Result
    
    n_train:
        训练区间结束的下标
    
    ratio_threshold
        回到估值范围内的点占整个区间的比例阈值,只有高于阈值的区间被视为成功回归估值,同时低于阈值的区间被视为偏离估值
    """
    # value: (T, 1)
    value = result.value
    band = result.band

    # GGS算法
    bps_v, _ = GGS(value.T, Kmax=10, lamb=1e-4)
    
    # 寻找第一个回到估值上的价值区间
    deviation = False
    deviation_start = None
    deviation_count = 0
    for start, end in zip(bps_v[-1][:-1], bps_v[-1][1:]):
        ratio = np.sum(np.abs(value[start:end]) <= band) / (end - start)
        if end > n_train:
            if ratio < ratio_threshold:
                # 偏离估值范围
                if not deviation:
                    # 首次偏离估值范围
                    deviation = True
                    deviation_start = start
                # 记录偏离点个数
                deviation_count += np.sum(np.abs(value[start:end]) > band)
            elif deviation and ratio >= ratio_threshold:
                # 偏离估值范围后首次回到估值范围
                return ValueTimeRange(
                    deviation_time=deviation_start,
                    return_time=start,
                    end_time=end,
                    deviation_ratio=deviation_count/(end-deviation_start),
                    return_ratio=ratio
                )
    
    if deviation:
        # 偏离估值范围后没有回归过
        return ValueTimeRange(
            deviation_time=deviation_start,
            deviation_ratio=deviation_count/(end-deviation_start)
        )
    else:
        # 没有偏离过估值范围
        return ValueTimeRange()
