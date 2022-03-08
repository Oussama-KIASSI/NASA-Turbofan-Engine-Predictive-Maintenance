import matplotlib.pyplot as _plt
import seaborn as _sns


def scatter_indicators(dataframe, title, indicators):
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
    fig, axes = _plt.subplots(indicators_count, 1, figsize=(14, 130))
    # create and adjust superior title
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=.975)
    # create palette
    p = _sns.color_palette("coolwarm", engines_count)
    # iterate over indicators
    for i, indicator in enumerate(indicators):
        # plot scatter plot of each indicator
        _sns.scatterplot(ax=axes[i], data=dataframe, x='Time', y=indicator, hue='Engine_no', palette=p, legend=False)


def line_indicators(dataframe, title, indicators):
    """scatter plot of indicators of each engine

    Args:
        dataframe: dataframe to analyze
        title: plot title
        indicators: indicators to visualize
    """
    # count of indicators
    indicators_count = len(indicators)
    # aggregate dataframe by Time (min, mean, max)
    df_grouped = dataframe.groupby('Time').agg(['min', 'mean', 'max'])
    # create subplots
    fig, axes = _plt.subplots(indicators_count, 1, figsize=(14, 130))
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
        _sns.lineplot(ax=axes[i], data=df_grouped_ind, x='Time', y='max', color='lightcoral', linewidth=.7)
        # plot lower edge
        _sns.lineplot(ax=axes[i], data=df_grouped_ind, x='Time', y='min', color='dodgerblue', linewidth=.7)
        # plot mean line
        _sns.lineplot(ax=axes[i], data=df_grouped_ind, x='Time', y='mean', color='darkred', linewidth=2)
        # set y axis title
        axes[i].set_ylabel(indicator)
