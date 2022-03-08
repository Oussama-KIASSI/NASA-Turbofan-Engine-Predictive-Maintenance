import pandas as _pd
import tqdm.notebook as _tqdm


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
