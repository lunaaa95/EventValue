import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Data:
    """Class for storing data

    x_treated: (T, 1)

    x_control: (T, n_control)

    label_treated: (1,)

    label_control: (n_control,)
    
    time: (T,)
    """
    x_treated: np.ndarray
    x_control: np.ndarray
    label_treated: np.ndarray
    label_control: np.ndarray
    time: np.ndarray

    def asdict(self):
        return asdict(self)


class Preprocess:
    @staticmethod
    def process(data, time_column:str, id_column:str, price_column:str, keepall=False) -> None:
        # 将面板数据转换为透视表
        data = data.pivot(
            index=time_column, columns=id_column, values=price_column
            )
        
        if keepall:
            selected_data = data
        else:
            # 选择价格数据缺失天数小于10%的股票
            selected_stock = data.isna().sum(axis=0) < 0.1 * len(data)
            selected_data = data[data.columns[selected_stock]]

        # 填充缺失的价格数据
        selected_data = selected_data.ffill().bfill()
        return selected_data
