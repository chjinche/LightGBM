import argparse
import logging
import pdb
import lightgbm as lgb
import numpy as np
import os



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--train-data-path',
    #     help='The file path for the training data.',
    # )
    # parser.add_argument(
    #     '--model-path',
    #     help='The output file path for the trained LightGBM model.',
    # )
    # args, _ = parser.parse_known_args()
    # train_data_path = args.train_data_path
    # model_path = args.model_path

    logger = logging.getLogger()

    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    train_data_path = "/mnt/chjinche/data/0903_mm_data/mm_70_no_str_no_header.csv"
    # test_data_path = "/mnt/chjinche/projects/LightGBM/tests/data/clean_input_parts_0"
    # scored_data_path = "/mnt/chjinche/projects/LightGBM/tests/data/pred_clean_input_parts_0"
    transform_file = "/mnt/chjinche/data/0903_mm_data/mm_ini_reformat"
    # header_file = "/mnt/chjinche/projects/LightGBM/tests/data/clean_header"
    header_file = "/mnt/chjinche/data/0903_mm_data/mm_no_str_header.csv"
    model_path = "/mnt/chjinche/projects/LightGBM/tests/data/trained_model"
    # score_tmp_transform_file = "tmp_transform"

    # logger.info('>>>>>LightGBM Train Module>>>>>')
    # # if os.path.isdir(train_data_path):
    # #     logger.info('[LightGBM Train] Directory "{}" is provided, use default file name "tp.data"'.format(train_data_path))
    # #     train_data_path = os.path.join(train_data_path, 'tp.data')
    train_data = lgb.Dataset(train_data_path, params={"transform_file": transform_file, "header_file": header_file})
    # train_data.construct()
    # # validation_data = train_data.create_valid(train_data_path)
    # # validation_data.construct()

    params = {
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'label':3,#'name:m_Rating',
    'query':0,#'name:m_QueryId',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'num_trees': 10,
    'num_leaves': 31,
    'header':False,
    'label_gain': "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100",
    # "training_metric":True,
    # "max_bin":1023
    }

    # params = {
    #     'boosting': 'gbdt',
    #     'learning_rate': 0.1,
    #     # 'lambda_l1': 0.1,
    #     # 'lambda_l2': 0.2,
    #     # 'max_depth': 4,
    #     'objective': 'multiclass',
    #     'metric': 'multi_logloss',
    #     'num_trees': 10,
    #     'num_leaves': 31,
    #     'num_class': 25,
    #     # 'num_class': np.max(train_data.get_label().astype(int)) + 1,
    #     # 'num_class': np.max(train_data.get_label().astype(int)) + 1,
    #     # Make sure the stable result with the same input
    #     'seed': 2021,
    #     'deterministic': 'true',
    #     'force_col_wise': 'true'
    # }

    bst = lgb.train(params, train_data, valid_sets=[train_data])

    # # truth_l = validation_data.get_label().astype(int)

    # # print(f'Correctness on training data: {np.sum(pred_l == truth_l)}/{len(pred_l)}')

    # if os.path.isdir(model_path):
    #     logger.info('[LightGBM Train] Directory "{}" is provided, use default file name "lgbm.model"'.format(model_path))
    #     model_path = os.path.join(model_path, 'lgbm.model')
    bst.save_model(model_path, transform_file=transform_file, header_file=header_file)
    logger.info('[LightGBM Train] Save model successfully!')
    logger.info('<<<<<LightGBM Train Module<<<<<')
    # logger.info('[LightGBM Train] Save model successfully!')
    # logger.info('<<<<<LightGBM Train Module<<<<<')
    # # pred_p = bst.predict(train_data_path)
    # # pred_l = np.argmax(pred_p, axis=1)

    # logger.info('>>>>>LightGBM Inference Module>>>>>')
    # bst = lgb.Booster(model_file=model_path, transform_file=score_tmp_transform_file)

    # pred_p = bst.predict(test_data_path, transform_file=score_tmp_transform_file, header_file=header_file)
    # # pred_l = np.argmax(pred_p, axis=1)
    # np.savetxt(scored_data_path, pred_p)
    # logger.info('<<<<<LightGBM Inference Module<<<<<')
