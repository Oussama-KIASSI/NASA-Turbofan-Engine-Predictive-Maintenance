import pandas as _pd
import os as _os
import re as _re
import tqdm.notebook as _tqdm
import pickle as _pkl


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


def missing_values_table(dataframes: dict[str, _pd.DataFrame]) -> dict[str, _pd.DataFrame]:
    """give info about missing data

    Args:
        dataframes: dictionary of dataframes

    Returns:
        dictionary of dataframes containing info about missing data in input dataframes
    """
    # initiate returned variable
    missing_info_dataframes = {}
    # iterate over dataframes
    for k in _tqdm.tqdm(dataframes):
        # shape of dataframe
        df_shape = dataframes[k].shape
        # count missing values in dataframe
        mis_val = dataframes[k].isnull().sum()
        # count percentage of missing values in dataframe
        mis_val_percent = 100 * dataframes[k].isnull().sum() / df_shape[0]
        # concatenate count and percentage
        mis_val_table = _pd.concat([mis_val, mis_val_percent], axis=1)
        # rename columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        # keep only columns with missing values and sort using percentage
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(2)
        # print recap message concerning missing values
        print(k + ' has ' + str(df_shape[1]) + ' columns.\n There are ' + str(mis_val_table_ren_columns.shape[0]) +
              ' columns that have missing values.')
        # add missing data info into dictionary
        missing_info_dataframes[k] = mis_val_table_ren_columns
    return missing_info_dataframes


def extract_rul(dataframes: dict[str, _pd.DataFrame]) -> dict[str, _pd.DataFrame]:
    """extract remaining useful lifetime (RUL) from dataset

    Args:
        dataframes: dictionary containing dataframes

    Returns:
        updated dictionary
    """
    # iterate over four sets
    for i in _tqdm.tqdm(range(1, 5)):
        # training data
        # since training data is about operation till failure, inverse time to get RUL. Differently
        # put, subtract time unit from maximum
        dataframes[f'train_FD00{i}']['RUL'] = dataframes[f'train_FD00{i}'].groupby('Engine_no')['Time'].transform(
            'max') - dataframes[f'train_FD00{i}']['Time']
        # test data
        # since RUL data is about RUL of each engine in each set, the last time unit should represent the
        # RUL from the RUL dataset.
        # extract Engine_no
        dataframes[f'RUL_FD00{i}']['Engine_no'] = dataframes[f'RUL_FD00{i}'].index + 1
        # merge test dataset with new RUL dataset
        dataframes[f'test_FD00{i}'] = dataframes[f'test_FD00{i}'].merge(dataframes[f'RUL_FD00{i}'], how='inner',
                                                                        on='Engine_no')
        # subtract time unit from maximum time plus RUL of last time unit
        dataframes[f'test_FD00{i}']['RUL'] += dataframes[f'test_FD00{i}'].groupby('Engine_no')['Time'].transform(
            'max') - dataframes[f'test_FD00{i}']['Time']
        # remove RUL dataframe since it will not be used after
        del dataframes[f'RUL_FD00{i}']
    return dataframes
