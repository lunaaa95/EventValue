from ts.data import Preprocess
from ts.value.data import TrainDataSet, TestDataSet
import ts.linear.model as linear
from ts.value.model import Value
from ts.value.metric import find_value_return


if __name__ == '__main__':
    n_train = 240
    n_test = 300

    import pandas as pd
    from tqdm import tqdm

    stocks = pd.read_parquet('./csmar/stock.parquet.gzip')

    raw_data = stocks[['Stkcd','Trddt','Clsprc','Adjprcnd']]
    raw_data = raw_data[(raw_data['Trddt']<'2023-01-01')&(raw_data['Trddt'] >= '2020-07-01')]

    ref_time = raw_data['Trddt'].unique()[n_train]
    end_time = raw_data['Trddt'].unique()[n_train+n_test]

    train_raw_data = raw_data[(raw_data['Trddt'] < ref_time)]
    test_raw_data = raw_data[(raw_data['Trddt'] < end_time)&(raw_data['Trddt'] >= ref_time)]

    interested_stock = []
    with open('/home/richardwu/projects/dayan/test/med.txt') as f:
        for line in f:
            interested_stock.append(line.strip().replace('\n', ''))

    from ts.data import Preprocess
    from ts.value.data import TrainDataSet, TestDataSet
    import ts.linear.model as linear
    from ts.value.model import Value
    from ts.value.metric import find_value_return
    from tqdm import tqdm

    train_ds = TrainDataSet(
        data=Preprocess.process(
            data=train_raw_data,
            time_column='Trddt',
            id_column='Stkcd',
            price_column='Clsprc'
        ),
        treated_stock_list=interested_stock
    )

    test_ds = TestDataSet(
        data=Preprocess.process(
            data=test_raw_data,
            time_column='Trddt',
            id_column='Stkcd',
            price_column='Clsprc',
            keepall=True
        )
    )

    for stock in tqdm(interested_stock):
        train_data = train_ds[stock]
        try:
            test_data = test_ds.infer_from_train(train_data)
        except AssertionError:
            continue
        sc = linear.SynthControl()
        value_model = Value(sc)
        train_result = value_model.train(train_data)
        test_result = value_model.infer(test_data)
        result = train_result.extend(test_result)
        vtr = find_value_return(result, n_train)
        if vtr.return_time:
            thx = (stock, result.time[vtr.deviation_time], result.time[vtr.return_time], result.time[min(vtr.end_time, len(result.time)-1)])
            with open('/home/richardwu/projects/dayan/test/med-dev-ret.txt', 'a') as f:
                f.write(str(thx)+'\n')
        elif vtr.deviation_time is None:
            thx = (stock,)
            with open('/home/richardwu/projects/dayan/test/med-no-dev.txt', 'a') as f:
                f.write(str(thx)+'\n')
        else:
            thx = (stock, result.time[min(vtr.deviation_time, len(result.time)-1)])
            with open('/home/richardwu/projects/dayan/test/med-dev-no.txt', 'a') as f:
                f.write(str(thx)+'\n')
