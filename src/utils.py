import pandas as _pd
import os as _os
import re as _re
import tqdm.notebook as _tqdm


def load_data(dataprefix1: str,
              dataschema1: list[str],
              dataschema2: list[str],
              sourcepath: str = '../data',
              datatype: str = '01_raw',
              dataext: str = '.txt',
              sep: str = ' ') -> dict[str, _pd.DataFrame]:
    """load data from specified path into dataframes

    Args:
        dataprefix1: first type dataset prefix
        dataschema1: first type dataset schema
        dataschema2: other type dataset schema
        sourcepath: data source parent folder
        datatype: data source child folder
        dataext: data extension
        sep: separator to use when loading

    Returns:
        dictionary of dataframes with names as keys and dataframes as values
    """
    # define data folder
    datapath = _os.path.join(sourcepath, datatype)
    # initiate returned variable
    dataframes = {}
    # iterate over files in data folder while visually displaying progress using tqdm
    for file in _tqdm.tqdm(_os.listdir(datapath)):
        # process only files with specified extension
        if file.endswith(dataext):
            filepath = _os.path.join(datapath, file)
            filename = _re.sub(dataext, '', file)
            # load first type dataset
            if file.startswith(dataprefix1):
                dataframes[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema1, usecols=dataschema1)
            # load other type dataset
            else:
                dataframes[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema2, usecols=dataschema2)
            shape = dataframes[filename].shape
            # info about loaded dataset
            print('\n' + '-' * 30 + '\n' + filename.center(30, '-') + '\n' + '-' * 30)
            print('Number of rows : %s ' % str(shape[0]))
            print('Number of columns : %s ' % str(shape[1]))
    return dataframes


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
        # put, substract time unit from maximum
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
        # substract time unit from maximum time plus RUL of last time unit
        dataframes[f'test_FD00{i}']['RUL'] += dataframes[f'test_FD00{i}'].groupby('Engine_no')['Time'].transform(
            'max') - dataframes[f'test_FD00{i}']['Time']
        # remove RUL dataframe since it will not be used after
        del dataframes[f'RUL_FD00{i}']
    return dataframes
