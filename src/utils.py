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
              sep: str = ' '):
    """

    Args:
        dataprefix1:
        dataschema1:
        dataschema2:
        sourcepath:
        datatype:
        dataext:
        sep:

    Returns:

    """
    datapath = _os.path.join(sourcepath, datatype)
    dataframes = {}
    for file in _tqdm.tqdm(_os.listdir(datapath)):
        if file.endswith(dataext):
            filepath = _os.path.join(datapath, file)
            filename = _re.sub(dataext, '', file)
            if file.startswith(dataprefix1):
                dataframes[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema1, usecols=dataschema1)
            else:
                dataframes[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema2, usecols=dataschema2)
            shape = dataframes[filename].shape
            print('\n' + '-' * 30 + '\n' + filename.center(30, '-') + '\n' + '-' * 30)
            print('Number of rows : %s ' % str(shape[0]))
            print('Number of columns : %s ' % str(shape[1]))
    return dataframes


def extract_rul(dataframes):
    for i in range(1, 5):
        dataframes[f'train_FD00{i}']['RUL'] = dataframes[f'train_FD00{i}'].groupby('Engine_no')['Time'].transform(
            'max') - dataframes[f'train_FD00{i}']['Time']
        dataframes[f'RUL_FD00{i}']['Engine_no'] = dataframes[f'RUL_FD00{i}'].index + 1
        dataframes[f'test_FD00{i}'] = dataframes[f'test_FD00{i}'].merge(dataframes[f'RUL_FD00{i}'], how='inner',
                                                                        on='Engine_no')
        dataframes[f'test_FD00{i}']['RUL'] += dataframes[f'test_FD00{i}'].groupby('Engine_no')['Time'].transform(
            'max') - dataframes[f'test_FD00{i}']['Time']
    return dataframes
