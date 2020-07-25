import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cbook as cbook
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_coeffs(coeffs, cols, axs = None, **kwargs):

    """
    Plot to show coefficients
    :param coeffs: list of coefficient values
    :param cols: list of features
    :param axs: matplotlib axis if using one from a preexisting plot
    :param **kwargs: other keywords
    """

    sns.set_style("whitegrid")

    if axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(kwargs.get("fig_x", 7),kwargs.get("fig_y", 7)))

    axs.scatter(coeffs, np.arange(len(coeffs))+1)
    axs.axvline(0.0, 0, 1, linestyle='--', color='k')

    axs.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))
    axs.set_ylim(kwargs.get("ymin", 0.5), kwargs.get("ymax", len(coeffs)+0.5))
    axs.set_xlabel(kwargs.get("xlabel",""), fontsize=kwargs.get("label_fontsize",16))
    axs.set_ylabel(kwargs.get("ylabel"," "), fontsize=kwargs.get("label_fontsize",16))
    axs.set_title(kwargs.get("title","Coeffs"), fontsize=kwargs.get("title_fontsize",18))
    axs.set_yticklabels(kwargs.get("y_tick_labels",['']+cols))
    axs.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    axs.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))

    try:
        return fig, axs
    except:
        return axs

def gen_coeff_box_stats(df, features, coeffs):

    """
    Create boxplot stats for coefficients
    :param df: pandas dataframe
    :param features: list
    :param coeffs: list
    """

    _df = df[features].copy()

    for i, ftr in enumerate(features):
        _df[ftr] *= coeffs[i]

    stats = cbook.boxplot_stats(_df.values, labels=_df.columns.to_list())#, bootstrap=10000)

    return _df, stats


def plot_box(stats, axs = None, **kwargs):

    """
    Coefficients box plot
    :param stats: cbook.boxplot_stats instance from MLLytics.regression.gen_coeff_box_stats
    :param axs: matplotlib axis if using one from a preexisting plot
    :param **kwargs: other keywords
    """

    sns.set_style("white")

    if axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(kwargs.get("fig_x", 7),kwargs.get("fig_y", 7)))


    axs.bxp(stats, vert=False)

    axs.axvline(0.0, 0, 1, linestyle='--', color='k')

    axs.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))
    axs.set_ylim(kwargs.get("ymin", None), kwargs.get("ymax", None))
    axs.set_xlabel(kwargs.get("xlabel",""), fontsize=kwargs.get("label_fontsize",16))
    axs.set_ylabel(kwargs.get("ylabel"," "), fontsize=kwargs.get("label_fontsize",16))
    axs.set_title(kwargs.get("title",None), fontsize=kwargs.get("title_fontsize",18))
    #axs.set_yticklabels(kwargs.get("y_tick_labels",['']+cols))
    axs.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    axs.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))

    try:
        return fig, axs
    except:
        return axs



def plot_resid(actual, predicted, axs = None, **kwargs):

    """
    Plot a partial dependency plot
    :param actual: array
    :param predicted: array
    :param axs: matplotlib axis if using one from a preexisting plot
    :param **kwargs: other keywords
    """

    sns.set_style("whitegrid")

    residuals = actual - predicted

    # definitions for the axes
    left, width = 0.1, 0.9
    bottom, height = 0.1, 0.9
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(6, 6))

    axs = fig.add_axes(rect_scatter)
    ax_histy = fig.add_axes(rect_histy, sharey=axs)

    ax_histy.tick_params(axis="y", labelleft=False, labelright=True)

    # the scatter plot:
    axs.scatter(predicted, residuals, alpha=0.25)
    axs.axhline(0.0, 0, 1, linestyle='--', color='k')

    # now determine nice limits by hand:
    binwidth = 0.25

    ymax = np.max(np.abs(residuals))
    lim = int(ymax)#(int(ymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)

    ax_histy.hist(residuals, bins=bins, orientation='horizontal')
    ax_histy.axhline(0.0, 0, 1, linestyle='--', color='k')

    ax_histy.set_xlabel('Count', fontsize=16)
    ax_histy.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    ax_histy.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))

    #if axs == None:
    #    fig, axs = plt.subplots(1, 1, figsize=(kwargs.get("fig_x", 7),kwargs.get("fig_y", 7)))

    axs.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))
    axs.set_ylim(kwargs.get("ymin", None), kwargs.get("ymax", None))
    axs.set_xlabel(kwargs.get("xlabel","Predicted"), fontsize=kwargs.get("label_fontsize",16))
    axs.set_ylabel(kwargs.get("ylabel","Residuals"), fontsize=kwargs.get("label_fontsize",16))
    #axs.set_title(kwargs.get("title","Coeffs"), fontsize=kwargs.get("title_fontsize",18))
    axs.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    axs.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))

    fig.suptitle('This is a somewhat long figure title', fontsize=18, y=1.05, x=0.67)


    #try:
    #    return fig, axs
    #except:
    #    return axs
    plt.show()
