import pandas as pd
import numpy as np
from dataclasses import dataclass

from .data import (
    TrainDataSet,
    TestDataSet
)
from ..data import Data
from ..linear.model import SynthControl


@dataclass
class Result:
    label: str
    time: np.ndarray
    value: np.ndarray
    band: float
    x: np.ndarray
    x_pred: np.ndarray

    def to_df(self):
        return pd.DataFrame(
            self.value,
            index=self.time,
            columns=[self.label]
        )
    
    def extend(self, other):
        return Result(
            label=self.label,
            time=np.concatenate((self.time, other.time)),
            value=np.vstack((self.value, other.value)),
            band=self.band,
            x=np.vstack((self.x, other.x)),
            x_pred = np.vstack((self.x_pred, other.x_pred))
        )


class Value:
    def __init__(self, model) -> None:
        self.model = model

    def train(self, train_data: Data):
        train_x_synth = self.model.generate(**train_data.asdict())

        # 计算价格偏离的价值
        # train_value为正对应高估,为负对应低估
        # train_value: (T, 1)
        train_value = train_data.x_treated - train_x_synth

        return Result(
            label=train_data.label_treated,
            time=train_data.time,
            value=train_value,
            band=self.model.band(),  # 按训练数据计算的估值波动区间
            x=train_data.x_treated,
            x_pred=train_x_synth
        )

    def infer(self, test_data: Data):
        test_x_synth = self.model.sample(x_control=test_data.x_control)

        # 估值同train
        # test_value: (T, 1)
        test_value = test_data.x_treated - test_x_synth

        return Result(
            label=test_data.label_treated,
            time=test_data.time,
            value=test_value,
            band=self.model.band(),  # 按训练数据计算的估值波动区间
            x=test_data.x_treated,
            x_pred=test_x_synth
        )
