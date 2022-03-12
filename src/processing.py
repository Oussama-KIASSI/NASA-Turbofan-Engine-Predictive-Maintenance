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
        dataframes[f'train_FD00{i}']['RUL'] = dataframes[f'train_FD00{i}'].groupby('Engine_no')['Cycle'].transform(
            'max') - dataframes[f'train_FD00{i}']['Cycle']
        # test data
        # since RUL data is about RUL of each engine in each set, the last time unit should represent the
        # RUL from the RUL dataset.
        # extract Engine_no
        dataframes[f'RUL_FD00{i}']['Engine_no'] = dataframes[f'RUL_FD00{i}'].index + 1
        # merge test dataset with new RUL dataset
        dataframes[f'test_FD00{i}'] = dataframes[f'test_FD00{i}'].merge(dataframes[f'RUL_FD00{i}'], how='inner',
                                                                        on='Engine_no')
        # subtract time unit from maximum time plus RUL of last time unit
        dataframes[f'test_FD00{i}']['RUL'] += dataframes[f'test_FD00{i}'].groupby('Engine_no')['Cycle'].transform(
            'max') - dataframes[f'test_FD00{i}']['Cycle']
        # remove RUL dataframe since it will not be used after
        del dataframes[f'RUL_FD00{i}']
    return dataframes


def add_DNf(dataframe: _pd.DataFrame) -> _pd.DataFrame:
    """add fan speed difference between demand and supply

    Args:
        dataframe: dataframe to process

    Returns:
        new dataframe with new DNf feature
    """
    # calculate difference and concatenate to dataframe
    return _pd.concat([dataframe, (dataframe['Nf_dmd'] - dataframe['Nf']).rename('DNf')], axis=1)


def add_DNRf(dataframe: _pd.DataFrame) -> _pd.DataFrame:
    """add corrected fan speed difference between demand and supply

    Args:
        dataframe: dataframe to process

    Returns:
        new dataframe with new DNRf feature
    """
    # calculate difference and concatenate to dataframe
    return _pd.concat([dataframe, (dataframe['PCNfR_dmd'] - dataframe['NRf']).rename('DNRf')], axis=1)


def add_p50(dataframe: _pd.DataFrame) -> _pd.DataFrame:
    """add outlet pressure at core nozzle

    Args:
        dataframe: dataframe to process

    Returns:
        new dataframe with new P50 feature
    """
    # calculate feature and concatenate to dataframe
    return _pd.concat([dataframe, (dataframe['P2'] * dataframe['epr']).rename('P50')], axis=1)


def add_expand_max(dataframe: _pd.DataFrame,
                   column) -> _pd.DataFrame:
    """add expanding window feature using maximum value in the specified column

    Args:
        dataframe: dataframe to process
        column: column to add its expanding window feature

    Returns:
        new dataframe with new expanding window feature
    """
    # calculate expanding window feature and concatenate to dataframe
    return _pd.concat([dataframe, dataframe.groupby('Engine_no')[column].expanding().max().rename(
        column + '_expandmax').reset_index(drop=True)], axis=1)


def feature_extraction(dataframe: _pd.DataFrame,
                        columns: list or tuple =
                        ('T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Ps30', 'P50')) -> _pd.DataFrame:
    """add new features to dataframe

    Args:
        dataframe: dataframe to process
        columns: columns to add their expanding window feature

    Returns:
        new dataframe with new extracted features
    """
    # add DNf feature
    dataframe_DNf = add_DNf(dataframe)
    # add DNf feature
    dataframe_DNRf = add_DNRf(dataframe_DNf)
    # add P50 feature
    dataframe_final = add_p50(dataframe_DNRf)
    # iterate over columns
    for column in columns:
        # add expanding window feature
        dataframe_final = add_expand_max(dataframe_final, column)
    return dataframe_final



