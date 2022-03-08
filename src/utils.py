import os as _os
import pickle as _pkl
import re as _re

import pandas as _pd
import tqdm.notebook as _tqdm


def load_data(data_prefix_1: str,
              data_schema_1: list[str],
              data_schema_2: list[str],
              source_path: str = '../data',
              data_type: str = '01_raw',
              data_ext: str = '.txt',
              sep: str = ' ') -> dict[str, _pd.DataFrame]:
    """load data from specified path into dataframes

    Args:
        data_prefix_1: first type dataset prefix
        data_schema_1: first type dataset schema
        data_schema_2: other type dataset schema
        source_path: data source parent folder
        data_type: data source child folder
        data_ext: data extension
        sep: separator to use when loading

    Returns:
        dictionary of dataframes with names as keys and dataframes as values
    """
    # define data folder
    data_path = _os.path.join(source_path, data_type)
    # initiate returned variable
    dataframes = {}
    # iterate over files in data folder while visually displaying progress using tqdm
    for file in _tqdm.tqdm(_os.listdir(data_path)):
        # process only files with specified extension
        if file.endswith(data_ext):
            file_path = _os.path.join(data_path, file)
            file_name = _re.sub(data_ext, '', file)
            # load first type dataset
            if file.startswith(data_prefix_1):
                dataframes[file_name] = _pd.read_csv(file_path, sep=sep, names=data_schema_1, usecols=data_schema_1)
            # load other type dataset
            else:
                dataframes[file_name] = _pd.read_csv(file_path, sep=sep, names=data_schema_2, usecols=data_schema_2)
            df_shape = dataframes[file_name].shape
            # info about loaded dataset
            print('\n' + '-' * 30 + '\n' + file_name.center(30, '-') + '\n' + '-' * 30)
            print('Number of rows : %s ' % str(df_shape[0]))
            print('Number of columns : %s ' % str(df_shape[1]))
            print('-' * 30 + '\n' + dataframes[file_name].dtypes.to_string())
    return dataframes


def save_data(dataframes: dict[str, _pd.DataFrame],
              source_path: str = '../data',
              data_type: str = '02_interim'):
    """save data in the specified path

    Args:
        dataframes: dictionary of dataframes to save
        source_path: data sink parent folder
        data_type: data sink child folder
    """
    # iterate over dataframes
    for k in _tqdm.tqdm(dataframes.keys()):
        # path where data will be stored
        file_path = _os.path.join(source_path, data_type, k)
        # save data to path in csv format
        dataframes[k].to_csv(file_path, index=False)


def save_scaler(scaler: object,
                scaler_store_path: str = '../models/02_scalers',
                scaler_type: str = 'MinMax'):
    """save scaler in the specified path

    Args:
        scaler: scaler to save
        scaler_store_path: scaler store folder
        scaler_type: scaler type/name
    """
    # path where scaler will be stored
    file_path = _os.path.join(scaler_store_path, scaler_type + 'Scaler.pkl')
    # save scaler as binary object
    with open(file_path, 'wb') as f:
        _pkl.dump(scaler, f)


def load_scaler(scaler_store_path: str = '../models/02_scalers',
                scaler_type: str = 'MinMax'):
    """return scaler from specified path

    Args:
        scaler_store_path: scaler store folder
        scaler_type: scaler type/name

    Returns:
        scaler
    """
    # path of scaler
    file_path = _os.path.join(scaler_store_path, scaler_type + 'Scaler.pkl')
    # read scaler as binary object
    with open(file_path, 'rb') as f:
        return _pkl.load(f)


def save_model(model: object,
               model_store_path: str = '../models/01_baseline',
               model_type: str = 'LinReg'):
    """save model to specified path

    Args:
        model: model to save
        model_store_path: model store folder
        model_type: model type/name
    """
    # path where model will be stored
    file_path = _os.path.join(model_store_path, model_type + '.pkl')
    # save model as binary object
    with open(file_path, 'wb') as f:
        _pkl.dump(model, f)


def load_model(model_store_path: str = '../models/01_baseline',
               model_type: str = 'LinReg'):
    """return model from specified path

    Args:
        model_store_path: model store folder
        model_type: model type/name

    Returns:
        model
    """
    # path of model
    file_path = _os.path.join(model_store_path, model_type + '.pkl')
    # read model as binary object
    with open(file_path, 'rb') as f:
        return _pkl.load(f)
