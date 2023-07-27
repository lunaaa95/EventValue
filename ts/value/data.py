import pandas as pd
import numpy as np

from ..data import Data


class TrainDataSet:
    def __init__(self, data: pd.DataFrame, treated_stock_list,
                 corrs:pd.DataFrame=None, n_control=300) -> None:
        """待估值股票和其他股票的训练数据

        data:
            待估值股票和其他股票的数据,不能含有na值
            type: DataFrame
            shape: (T, n)

        treated_stock_list:
            待估值股票名称
            type: list-like
        
        n_control:
            每个待估值股票筛选出的其他股票数量
        """
        self.data = data
        self.treated_stock_list = pd.Index(treated_stock_list)
        self.n_control = n_control

        # 确保数据中含有待估值股票
        treated_contained = self.treated_stock_list.isin(data.columns)
        assert treated_contained.all(), f'{self.treated_stock_list[~treated_contained]} not in data'

        # 准备不含待估值股票的名称列表
        self.control_stock_list = data.columns.drop(self.treated_stock_list, errors='ignore')

        # 按相关性筛选其他股票
        self.corrs = corrs
        self.control_indices = self._filter_control()

    def _filter_control(self, divide_by_first=False):
        """按价格相关性过滤相关性很小的其他股票
        """
        data = self.data
        if divide_by_first:
            data = data / data.iloc[0]
        if self.corrs is None:
            # 计算待估值股票和其他股票的价格的相关性
            self.corrs = data.corr().loc[self.treated_stock_list]
            # 将待估值股票从相关性矩阵的列中剔除
            self.corrs.drop(columns=self.treated_stock_list, errors='ignore', inplace=True)
        
        # 按相关性排序
        select_indices = np.argsort(self.corrs.abs(), axis=1)
        # 取最大的(对应select_indices最末)m个
        select_indices = select_indices[:, -self.n_control:]

        return select_indices

    def __getitem__(self, item: int|str):
        """获取一只待估值股票的数据
        """
        if isinstance(item, str):
            idx = self.treated_stock_list.get_loc(item)
        else:
            idx = item
            item = self.treated_stock_list[idx]
        x_treated = self.data[item]
        
        select_stock = self.control_stock_list[self.control_indices[idx]]
        x_control = self.data[select_stock]
        
        return Data(
            x_treated=np.array(x_treated).reshape(-1, 1),
            x_control=np.array(x_control),
            label_treated=np.array([item]),
            label_control=np.array(select_stock),
            time=np.array(x_treated.index)
        )


class TestDataSet:
    def __init__(self, data: pd.DataFrame) -> None:
        """待估值股票和其他股票的预测数据

        data:
            待估值股票和其他股票的数据,不能含有na值
            type: DataFrame
            shape: (T, n)
        """
        self.data = data
    
    def infer_from_train(self, train_data: Data):
        """从训练数据推断对应预测数据

        train_data:
            对应训练数据
            type: data.Data
        """
        treated_contained = pd.Index(train_data.label_treated).isin(self.data.columns)
        assert treated_contained.all(), f'{train_data.label_treated[~treated_contained]} not in data'
        control_contained = pd.Index(train_data.label_control).isin(self.data.columns)
        assert control_contained.all(), f'{train_data.label_control[~control_contained]} not in data'
        return Data(
            x_treated=np.array(self.data[train_data.label_treated]).reshape(-1, 1),
            x_control=np.array(self.data[train_data.label_control]),
            label_treated=train_data.label_treated,
            label_control=train_data.label_control,
            time=np.array(self.data.index)
        )