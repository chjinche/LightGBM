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

    train_data_path = "/mnt/chjinche/projects/LightGBM/tests/data/clean_input_parts_0_full"
    test_data_path = "/mnt/chjinche/projects/LightGBM/tests/data/clean_input_parts_0"
    scored_data_path = "./pred_clean_input_parts_0"
    transform_file = "/mnt/chjinche/projects/LightGBM/tests/data/SmoothedTrainInputIni"
    header_file = "/mnt/chjinche/projects/LightGBM/tests/data/clean_header"
    model_path = "./trained_model"

    # logger.info('>>>>>LightGBM Train Module>>>>>')
    # # if os.path.isdir(train_data_path):
    # #     logger.info('[LightGBM Train] Directory "{}" is provided, use default file name "tp.data"'.format(train_data_path))
    # #     train_data_path = os.path.join(train_data_path, 'tp.data')
    # train_data = lgb.Dataset(train_data_path, params={"transform_file": transform_file, "header_file": header_file})
    # # train_data.construct()
    # # validation_data = train_data.create_valid(train_data_path)
    # # validation_data.construct()

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

    # bst = lgb.train(params, train_data) #, valid_sets=[validation_data])

    # # # truth_l = validation_data.get_label().astype(int)

    # # # print(f'Correctness on training data: {np.sum(pred_l == truth_l)}/{len(pred_l)}')

    # # if os.path.isdir(model_path):
    # #     logger.info('[LightGBM Train] Directory "{}" is provided, use default file name "lgbm.model"'.format(model_path))
    # #     model_path = os.path.join(model_path, 'lgbm.model')
    # bst.save_model(model_path, transform_file=transform_file, header_file=header_file)
    # logger.info('[LightGBM Train] Save model successfully!')
    # logger.info('<<<<<LightGBM Train Module<<<<<')
    # # pred_p = bst.predict(train_data_path)
    # # pred_l = np.argmax(pred_p, axis=1)

    logger.info('>>>>>LightGBM Inference Module>>>>>')
    bst = lgb.Booster(model_file=model_path)

    pred_p = bst.predict(test_data_path)
    # pred_l = np.argmax(pred_p, axis=1)
    np.savetxt(scored_data_path, pred_p)
    logger.info('<<<<<LightGBM Inference Module<<<<<')
