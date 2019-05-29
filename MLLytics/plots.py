import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix

from MLLytics import cluster_correlation_matrix
from matplotlib import cm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import auc
from scipy import interp


def plot_roc_auc(fpr,tpr,threshold,youden=None):

    """
    Plot a roc_auc_curve
    :param fpr: Array of FPRs 
    :param fpr: Array of TPRs 
    :param fpr: Array of Thresholds
    :param youden: List (value, threshold)
	
    """
  
    sns.set_style("whitegrid")
    plt.figure(figsize=(7,7))

    plt.plot(fpr,tpr,c='k', zorder=1, linestyle='-' )
    plt.plot([-0.2,1.2],[-0.2,1.2], zorder=1, linestyle='--', color='k')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.scatter(fpr,tpr, c=threshold, cmap = cm.viridis, edgecolors='k', linewidth=1.5, marker='o',
                linestyle='-',zorder=2 )
    
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    
    get_auc = auc(fpr,tpr)
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    
    if youden is not None:
        plt.text(0.6,0.25,'AUC = '+str(round(get_auc,2))+'\nyouden_J_statistic = '+str(round(youden[0],2))+'\nthresehold = '+str(round(youden[1],2)), fontsize=12, bbox=props)
    else:
        plt.text(0.6,0.25,'AUC = '+str(round(get_auc,2)), fontsize=12, bbox=props)
        

    
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)

    cbaxes = inset_axes(plt.gca(), width="30%", height="3%", loc= 'lower right',bbox_to_anchor=(-0.02, 0.05, 1, 1),
                       bbox_transform=plt.gca().transAxes,
                       borderpad=0) 
    plt.colorbar(cax=cbaxes, orientation='horizontal')
    plt.text(0.33, 1.4, 'Threshold', fontsize=10)
    plt.show()

def plot_rp(recall,prec,threshold):

    """
    Plot a recall-precision curve
    :param fpr: Array of recall values
    :param fpr: Array of precision values
    :param fpr: Array of thresholds
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(7,7))

    plt.plot(recall,prec,c='k', zorder=1, linestyle='-' )
    #plt.plot([-0.2,1.2],[-0.2,1.2], zorder=1, linestyle='--', color='k')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.scatter(recall,prec, c=threshold, cmap = cm.viridis, edgecolors='k', linewidth=1.5, marker='o',
                linestyle='-',zorder=2 )
    #plt.plot(C,B, c='k',linewidth=1.5, marker='o', linestyle='-' )

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='k', alpha=0.5, linestyle='-.', zorder=1)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.85, y[45] - 0.05))

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
        
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)

    cbaxes = inset_axes(plt.gca(), width="30%", height="3%", loc= 'lower left',bbox_to_anchor=(0.05, 0.05, 1, 1),
                       bbox_transform=plt.gca().transAxes,
                       borderpad=0) 
    plt.colorbar(cax=cbaxes, orientation='horizontal')
    plt.text(0.33, 1.4, 'Threshold', fontsize=10)
    plt.show()



def reliability_curve(y_true, y_score, bins=25, normalize=False):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities whould be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels (0 or 1).

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=False
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is mapped onto 0 and the largest one
        onto 1.


    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.


    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        
        if len(y_score[bin_idx] > 0): #This suppresses warnings when calculating mean of empty arrays
            y_score_bin_mean[i] = y_score[bin_idx].mean() 
            empirical_prob_pos[i] = y_true[bin_idx].mean()
        else:
            y_score_bin_mean[i] = np.nan
            empirical_prob_pos[i] = np.nan
        
    return y_score_bin_mean, empirical_prob_pos

def plot_reliability_curve(prob, label, method='Model', bins=25):
    
    viridis = cm.viridis

    prob_bin_mean, bin_positive_frac = reliability_curve(label, prob, bins=bins, normalize=False)

    fig = plt.figure(0, figsize=(8, 8))
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], color = 'k', label="Perfect", linestyle='--', zorder=0)

    scores_not_nan = np.logical_not(np.isnan(bin_positive_frac))
    plt.plot(prob_bin_mean[scores_not_nan],bin_positive_frac[scores_not_nan], label=method, color = viridis(0.0),zorder=1)
    plt.scatter(prob_bin_mean[scores_not_nan],bin_positive_frac[scores_not_nan], color = viridis(0.45),edgecolors=viridis(0.0), zorder=2)
    plt.ylabel("Fraction of Positives", fontsize=16)
    plt.legend(loc=0)

    plt.subplot2grid((3, 1), (2, 0))
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    plt.hist(prob, range=(0, 1), bins=bins, label=method,
                 histtype="step", lw=2,color = viridis(0.0))
    plt.xlabel("Predicted Probability", fontsize=16)
    plt.ylabel("Bin Count", fontsize=16)

    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)


    plt.legend(loc='upper center', ncol=2)

    
def plot_confusion_matrix(prob, label, label_names, threshold=0.5,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pred = np.zeros(len(label))
    pred[prob>threshold] = 1
    
    cm = confusion_matrix(label,pred)#,labels = [0,1,2])
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    sns.set_style("white")
#    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, label_names, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.show()
	
def plot_corr_matrix_triangle(_corr, cmap=cm.coolwarm):

    """
    Plot a traingular correlation matrix
    :param _corr: Correlations from df.corr()
    """

    ## corr matrix triangle
    sns.set(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 7)) 
    
    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #cmap = cm.PiYG
    
    sns.heatmap(_corr, mask=mask, cmap=cmap, square=True, cbar_kws={"shrink": .75}, vmin=-1., vmax=1.)
    ax.set_title('Correlation Matrix', fontsize=14)            
	
#def plot_corr_hist(corr):
#
#    """
#    Plot a histogram of correlation values
#    :param corr: A list of ClassMetrics objects for different CV folds
#    """
#
#    plt.hist(np.array(list(a.values())), bins=10)
#    plt.xlim([-1,1])
	
def plot_cluster_corr(df):

    """
    Plot a clustered correlation matrix
    :param df: Pandas dataframe 
    """
    
    x, clus_corr = cluster_correlation_matrix(df)
    
    sns.set(style="white")
    
    corr = clus_corr
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(9, 7))
    #cmap = sns.diverging_palette(20, 10, as_cmap=True)
    cmap = cm.BuPu
    
    sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={"shrink": .75}, vmin=0., vmax=1.)
    ax.set_title('Clustered Correlation Matrix', fontsize=14)

    j = 0
    k = 0
    for i in range(0,len(set(x.values()))):
        z = sum(value == i for value in x.values())
        k+=z
        l=k-z
     
        if i == 0:
            lt_a = lt_c = 3
            lt_b = lt_d = 2
        elif i == (len(set(x.values())) - 1):
            lt_a = lt_c = 2
            lt_b = lt_d = 5        
        else:
            lt_a = lt_b = lt_c = lt_d = 2
       
        ax.hlines(l,l,k, color='k', linewidth=lt_a)
        ax.hlines(k,l,k, color='k', linewidth=lt_b)
        ax.vlines(l,l,k, color='k', linewidth=lt_c)
        ax.vlines(k,l,k, color='k', linewidth=lt_d)    

def plot_ftr_importance(cols, imps, n=None):
    
    d_keys, d_values = ftr_importance(cols, imps)
    
    if n is not None:
        d_keys = d_keys[:n]
        d_values = d_values[:n]
        
    sns.set(style="whitegrid")

    plt.figure(figsize=(9,6))
    ax = sns.barplot(y=d_keys, x=d_values, orient = 'h',palette=("viridis"))

    ax.set_title('Top 10 Features',fontsize=20)
    ax.set_xlabel('Importance (%)', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=14)
    #fig = plt.gcf()
    #plt.xticks(rotation=90)
	

def plot_roc_auc_cv(folds: list, label='Fold', plot_averages = True):
    
    """
    Plot a ROC-AUC curve on many folds
    :param folds: A list of ClassMetrics objects for different CV folds
    """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(7,7))
    
    k = len(folds)
    tprs = []
    mean_fpr = np.linspace(0, 1, 101)
    _auc = np.empty(0)
    
    for i in range(0,k):
        tprs.append(interp(mean_fpr,folds[i].fpr[::-1], folds[i].tpr[::-1]))
        tprs[-1][0] = 0.0
        _auc=np.append(_auc, auc(folds[i].fpr, folds[i].tpr))
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot([-0.2,1.2],[-0.2,1.2], zorder=1, linestyle='--', color='k')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.title("ROC-AUC with CV", fontsize=18)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        
    #print(_auc)
    mean_auc = np.mean(np.array(_auc))
    std_auc = np.std(np.array(_auc))
	
	
    
    for i in range(0,k):

        plt.plot(folds[i].fpr,folds[i].tpr, zorder=1, linestyle='-',
                label=label+r' %d : AUC = %0.2f'%(i,_auc[i]))
    
    if plot_averages == True:
	    plt.plot(mean_fpr, mean_tpr, color='k', linewidth=2, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
	    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
			label=r'$\pm$ 1 std. dev.')
				 
				 
				 
    
    plt.legend(fontsize = 12)
    
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14)
    



def plot_recall_precision_cv(folds: list, label='Fold', plot_averages = True):
    
    """
    Plot a recall precision curve on many folds
    :param folds: A list of ClassMetrics objects for different CV folds
    """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(7,7))
    
    k = len(folds)
    pres = []
    mean_recalls = np.linspace(0, 1, 101)
    
    for i in range(0,k):
        pres.append(interp(mean_recalls,folds[i].recall[::-1], folds[i].prec[::-1]))
        #pres[-1][0] = 0.0
        #_auc=np.append(_auc, auc(folds[i].fpr, folds[i].tpr))
        
    mean_prec = np.mean(pres, axis=0)
    #mean_prec[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("RP curve with CV", fontsize=18)

    
    std_prec = np.std(pres, axis=0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)

        
    #print(_auc)
    #mean_auc = np.mean(np.array(_auc))
    #std_auc = np.std(np.array(_auc))
    
    for i in range(0,k):
        plt.plot(folds[i].recall,folds[i].prec, zorder=1, linestyle='-',label=label+r' %d'%(i))
    
    if plot_averages == True:
	    plt.plot(mean_recalls, mean_prec, color='k',label=r'Mean', linewidth=2)
	    plt.fill_between(mean_recalls, prec_lower, prec_upper, color='grey', alpha=.2,
			label=r'$\pm$ 1 std. dev.')
    
    plt.legend(fontsize=14)
    
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14)
    
    
        #plt.scatter(folds[i].fpr,folds[i].tpr, cmap = cm.viridis, edgecolors='k',
        #            linewidth=1.5, marker='o',linestyle='-',zorder=2 )
