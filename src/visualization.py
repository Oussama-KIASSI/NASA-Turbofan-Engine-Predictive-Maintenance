import matplotlib.pyplot as _plt
import seaborn as _sns
import pandas as _pd
import numpy as _np


def scatter_indicators(dataframe: _pd.DataFrame,
                       title: str,
                       indicators: list[str]):
    """scatter plot of indicators of each engine

    Args:
        dataframe: dataframe to analyze
        title: plot title
        indicators: indicators to visualize
    """
    # count of indicators
    indicators_count = len(indicators)
    # count of engines in dataframe
    engines_count = dataframe.Engine_no.nunique()
    # create subplots
    fig, axes = _plt.subplots(nrows=indicators_count, ncols=1, figsize=(14, 130))
    # create and adjust superior title
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=.975)
    # create palette
    p = _sns.color_palette("coolwarm", engines_count)
    # iterate over indicators
    for i, indicator in enumerate(indicators):
        # plot scatter plot of each indicator
        _sns.scatterplot(data=dataframe, ax=axes[i], x='RUL', y=indicator, hue='Engine_no', palette=p, legend=False)
        # invert x-axis
        axes[i].invert_xaxis()


def displot_indicators(dataframe: _pd.DataFrame,
                       title: str,
                       indicators: list[str]):
    """distribution plot of indicators of each engine

    Args:
        dataframe: dataframe to analyze
        title: plot title
        indicators: indicators to visualize
    """
    # count of indicators
    indicators_count = len(indicators)
    # create subplots
    fig, axes = _plt.subplots(nrows=indicators_count, ncols=1, figsize=(12, 130))
    # create and adjust superior title
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=.975)
    # iterate over indicators
    for i, indicator in enumerate(indicators):
        # plot scatter plot of each indicator
        _sns.histplot(data=dataframe, ax=axes[i], x=indicator, color='darkred')


def line_indicators(dataframe: _pd.DataFrame,
                    title: str,
                    indicators: list[str]):
    """scatter plot of indicators of each engine

    Args:
        dataframe: dataframe to analyze
        title: plot title
        indicators: indicators to visualize
    """
    # count of indicators
    indicators_count = len(indicators)
    # aggregate dataframe by Time (min, mean, max)
    df_grouped = dataframe.groupby(['RUL']).agg(['min', 'mean', 'max'])
    # create subplots
    fig, axes = _plt.subplots(nrows=indicators_count, ncols=1, figsize=(14, 130))
    # create and adjust superior title
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=.975)
    # iterate over indicators
    for i, indicator in enumerate(indicators):
        # choose indicator
        df_grouped_ind = df_grouped[indicator]
        # plot filling between max and min
        axes[i].fill_between(df_grouped_ind.index, df_grouped_ind['min'], df_grouped_ind['max'], color='gray',
                             alpha=0.1)
        # plot upper edge
        _sns.lineplot(data=df_grouped_ind, ax=axes[i], x='RUL', y='max', color='lightcoral', linewidth=.7)
        # plot lower edge
        _sns.lineplot(data=df_grouped_ind, ax=axes[i], x='RUL', y='min', color='dodgerblue', linewidth=.7)
        # plot mean line
        _sns.lineplot(data=df_grouped_ind, ax=axes[i], x='RUL', y='mean', color='darkred', linewidth=2)
        # set y axis title
        axes[i].set_ylabel(indicator)
        # invert x-axis
        axes[i].invert_xaxis()


def correlation_heatmap(dataframe: _pd.DataFrame,
                        indicators: list[str]):
    """plot correlation heatmap between indicators and RUL

    Args:
        dataframe: dataframe to analyze
        indicators: indicators to analyze
    """
    # add RUL columns to indicators to compute correlation
    indicators_ = indicators + ['RUL']
    # compute correlation of specified columns and RUL
    corr_matrix = dataframe.loc[:, indicators_].corr()
    # create mask to show only lower half of heatmap
    mask = _np.zeros_like(a=corr_matrix)
    mask[_np.triu_indices_from(arr=mask, k=1)] = 1
    # plot heatmap of correlation
    _plt.figure(figsize=(15, 15))
    _sns.heatmap(data=corr_matrix, vmax=.8, annot=True, mask=mask, cmap="coolwarm", square=False)
    _plt.show()
    # print textual analysis
    # iterate over correlation matrix columns
    for i, ind in enumerate(indicators_):
        # iterate over elements from lower triangle in the heatmap
        for j in range(i):
            # correlation value
            corr_ij = corr_matrix.iloc[i, j]
            # print interpretation
            if corr_ij > .8:
                print(f'• {ind} is strongly positively correlated with {indicators_[j]} = %.2f' % corr_ij)
            elif corr_ij < -.8:
                print(f'• {ind} is strongly negatively correlated with {indicators_[j]} = %.2f' % corr_ij)
