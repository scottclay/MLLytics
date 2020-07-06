import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def make_pdp(df, feature, model, type='classification'):

    """
    Computes partial dependency plot values for a given feature.
    :param df: pandas dataframe
    :param feature: string
    :param model: sci-kit learn model instance
    :param type: string. classification or regression
    """

    min_val = df[feature].min()
    max_val = df[feature].max()

    values = np.arange(min_val, max_val, (max_val - min_val)* 0.01)

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

    return va, li


def plot_pdp(feature, va, li, type='classification'):

    """
    Plot a partial dependency plot
    :param feature: string
    :param va: array
    :param li: array
    :param type: string
    """
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(7,7))

    plt.plot(va,li,c='k', zorder=1, linestyle='-' )
    if type=='classification':
        plt.plot([min(va), max(va)],[0.,0.], linestyle='--')
    plt.ylabel("Partial Dependence", fontsize=16)
    plt.xlabel(feature, fontsize=16)

    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)

    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)

    plt.title("Partial Dependency Plot", fontsize=18)
    plt.show()
