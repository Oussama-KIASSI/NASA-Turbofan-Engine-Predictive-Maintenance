import IPython.core.display_functions as _ipdisplay
import pandas as _pd
import tqdm.notebook as _tqdm


def missing_values(dataframes: dict[str, _pd.DataFrame]) -> dict[str, _pd.DataFrame]:
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
        print('-' * 50 + '\n' + '-' * 50)
        print('\n' + k + ' has ' + str(df_shape[1]) + ' columns.\nThere are ' + str(mis_val_table_ren_columns.shape[0])
              + ' columns that have missing values.')
        _ipdisplay.display(mis_val_table_ren_columns)
        # add missing data info into dictionary
        missing_info_dataframes[k] = mis_val_table_ren_columns
    return missing_info_dataframes


def duplicate_rows(dataframes: dict[str, _pd.DataFrame]) -> dict[str, _pd.DataFrame]:
    """give info about duplicated rows

    Args:
        dataframes: dictionary of dataframes

    Returns:
        dictionary of dataframes containing info about duplicated rows in input dataframes
    """
    # initiate returned variable
    duplicate_rows_dataframes = {}
    # iterate over dataframes
    for k in _tqdm.tqdm(dataframes):
        # check for duplicated rows
        duplicate_rows_df = dataframes[k][dataframes[k].duplicated()]
        # print duplication info
        print('-' * 50 + '\n' + '-' * 50)
        print('\n' + k + ' has ' + str(duplicate_rows_df.shape[0]) + ' duplicated rows.\n')
        # add duplicated data to dictionary
        duplicate_rows_dataframes[k] = duplicate_rows_df
    return duplicate_rows_dataframes
