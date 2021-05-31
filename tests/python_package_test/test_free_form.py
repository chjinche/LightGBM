from pathlib import Path
import lightgbm as lgb
from lightgbm.basic import FreeForm
import pytest
import pandas as pd


@pytest.fixture
def init_transform():
    input_file = Path(__file__).parent / 'training_input.ini'
    with open(input_file, 'r') as fin:
        for line in fin:
            if line.startswith('Line'):
                expression = line.split('(')[1].split(')')[0]
                op = expression.split(' ')[0]
                fea_names = expression.split(' ')[1:]
                return FreeForm(op=op, fea_names=fea_names)


@pytest.fixture
def input_df():
    return pd.DataFrame({'a': [1, 2, 4], 'b': [0, 3, 2]})


def test_support_free_form(input_df, init_transform):
    train_data = lgb.Dataset(input_df, label=[1, 1, 0], transform=init_transform)
    params = {
        "objective": "binary",
        "metric": "auc",
        "min_data": 1,
        "num_leaves": 2,
        "verbose": -1,
        "num_threads": 1,
        "max_bin": 255,
        "gpu_use_dp": True
    }
    # gbm = lgb.train(params, train_data, num_boost_round=20)
    bst = lgb.Booster(params, train_data)
    for i in range(5):
        bst.update()
        bst.eval_train()
    assert train_data.get_feature_name() == ['a', 'b', 'max_a_b']
    assert bst.current_iteration() == 1
