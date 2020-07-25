import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def make_pdp(df, feature, model, type='classification', quantiles=[0.05, 0.95]):

    """
    Computes partial dependency plot values for a given feature.
    :param df: pandas dataframe
    :param feature: string
    :param model: sci-kit learn model instance
    :param type: string. classification or regression
    :param quantiles: list. min max quantiles to use to exclude extreme values
    """

    min_val = df[[feature]].quantile(q=quantiles[0]).values[0]
    max_val = df[[feature]].quantile(q=quantiles[1]).values[0]

    values = np.arange(min_val, max_val, (max_val - min_val)* 0.01)

    qtls = {}
    for i in np.arange(0.1,1.0,0.1):
        qtls[np.round(i,1)] = df[[feature]].quantile(q=i).values[0]

    li = []
    va = []

    if type=='classification':
        for i in values:
            _df = df.copy()

            _df[feature].values[:] = i

            output = model.predict_proba(_df)[:, 1]

            vote_1 = len(output[output >= 0.5])
            vote_2 = len(output[output < 0.5])

            output = np.log(vote_1) - 0.5*(np.log(vote_1) + np.log(vote_2))

            avg_output = output.mean()

            li.append(avg_output)
            va.append(i)

    elif type=='regression':
        for i in values:
            _df = df.copy()

            _df[feature].values[:] = i

            output = model.predict(_df)

            avg_output = output.mean()

            li.append(avg_output)
            va.append(i)

    return va, li, qtls


def plot_pdp(feature, va, li, type='classification', quantiles = None, norm=False, axs = None, **kwargs):

    """
    Plot a partial dependency plot
    :param feature: string
    :param va: array
    :param li: array
    :param type: string
    :param quantiles: list of quantile values to plot
    :param norm: boolean. to normalise data so mean value = 0
    :param axs: matplotlib axis if using one from a preexisting plot
    :param **kwargs: other keywords
    """

    sns.set_style("whitegrid")

    if axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(7,7))

    if norm == True:
        li=np.array(li)
        li-=li.mean()

    axs.plot(va,li,c='k', zorder=1, linestyle='-' )

    axs.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))
    axs.set_ylim(kwargs.get("ymin", None), kwargs.get("ymax", None))
    axs.set_xlabel(kwargs.get("xlabel",feature), fontsize=kwargs.get("label_fontsize",16))
    axs.set_ylabel(kwargs.get("ylabel","Partial Dependence"), fontsize=kwargs.get("label_fontsize",16))
    axs.set_title(kwargs.get("title","Partial Dependency Plot"), fontsize=kwargs.get("title_fontsize",18))
    axs.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    axs.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))

    if quantiles is not None:
        for q in quantiles.keys():
            axs.axvline(quantiles[q], 0, 0.05)

    try:
        return fig, axs
    except:
        return axs


def make_ice(df, feature, model, type='classification', quantiles=[0.05, 0.95]):

    """
    Computes partial dependency plot values for a given feature.
    :param df: pandas dataframe
    :param feature: string
    :param model: sci-kit learn model instance
    :param type: string. classification or regression
    :param quantiles: list. min max quantiles to use to exclude extreme values
    """

    df_sample = df.sample(50)

    min_val = df[[feature]].quantile(q=quantiles[0]).values[0]
    max_val = df[[feature]].quantile(q=quantiles[1]).values[0]
    values = np.arange(min_val, max_val, (max_val - min_val)* 0.01)
    qtls = {}
    for i in np.arange(0.1,1.0,0.1):
        qtls[np.round(i,1)] = df[[feature]].quantile(q=i).values[0]

    li = []
    va = []

    if type=='classification':
        for i in values:
            _df = df.copy()

            _df[feature].values[:] = i

            output = model.predict_proba(_df)[:, 1]

            vote_1 = len(output[output >= 0.5])
            vote_2 = len(output[output < 0.5])

            output = np.log(vote_1) - 0.5*(np.log(vote_1) + np.log(vote_2))

            avg_output = output.mean()

            li.append(avg_output)
            va.append(i)

    elif type=='regression':
        for i in values:
            _df = df_sample.copy()

            _df[feature].values[:] = i

            output = model.predict(_df)

            #avg_output = output.mean()

            li.append(output)
            va.append(i)

    return va, li, qtls



def plot_ice(feature, va, li, many_li, type='classification', quantiles = None, norm=False, axs = None, **kwargs):

    """
    Plot a partial dependency plot
    :param feature: string
    :param va: array
    :param li: array
    :param many_li: pandas df
    :param type: string
    :param quantiles: list of quantile values to plot
    :param norm: boolean. to normalise data so mean value = 0
    :param axs: matplotlib axis if using one from a preexisting plot
    :param **kwargs: other keywords
    """

    sns.set_style("whitegrid")

    if axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(7,7))

    if norm == True:
        many_li = many_li.apply(lambda x: x-x[0], axis=0)
        li=np.array(li)
        li-=li[0]
        #if many_li is not None:
        #    many_li = many_li - li.mean()


    axs.plot(va,li,c='k', zorder=1, linestyle='-' )

    #axs.set_xlim(kwargs.get("xmin", None), kwargs.get("xmax", None))
    #axs.set_ylim(kwargs.get("ymin", None), kwargs.get("ymax", None))
    axs.set_xlabel(kwargs.get("xlabel",feature), fontsize=kwargs.get("label_fontsize",16))
    axs.set_ylabel(kwargs.get("ylabel","Partial Dependence"), fontsize=kwargs.get("label_fontsize",16))
    axs.set_title(kwargs.get("title","Partial Dependency Plot"), fontsize=kwargs.get("title_fontsize",18))
    axs.tick_params(axis='both', which='major', labelsize=kwargs.get("major_tick_fontsize",15))
    axs.tick_params(axis='both', which='minor', labelsize=kwargs.get("minor_tick_fontsize",15))


    for i in range(0,len(many_li.columns)):
        axs.plot(va, many_li[i] - many_li[i][0], color='blue', alpha=0.2)

    if quantiles is not None:
        for q in quantiles.keys():
            axs.axvline(quantiles[q], 0, 0.05)


    try:
        return fig, axs
    except:
        return axs
