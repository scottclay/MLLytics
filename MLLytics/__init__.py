import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from collections import OrderedDict
from matplotlib import cm

from sklearn.metrics import auc

class ClassMetrics():

    def __init__(self, prob,label, thres_bins=110):
        
    
        self.threshold = np.empty(0)
        self.tpr = np.empty(0)
        self.fpr = np.empty(0)
        self.tnr = np.empty(0)
        self.prec = np.empty(0)
        self.recall = np.empty(0)
        self.acc = np.empty(0)
        
        self.tp = np.empty(0)
        self.tn = np.empty(0)
        self.fp = np.empty(0)
        self.fn = np.empty(0)
        
        for thres in range(0,101,1):
            th = np.zeros(len(label))
            th[(prob>(thres/100.))] = 1

            _tp = len(th[(label==1)&(th==1)])
            _tn = len(th[(label==0)&(th==0)])
            _fp = len(th[(label==0)&(th==1)])
            _fn = len(th[(label==1)&(th==0)])

            _tpr = _tp / (_tp + _fn)
            _fpr = _fp / (_fp + _tn)
            _tnr = _tn / (_tn + _fp)

            try:
                _precision = _tp / (_tp + _fp)
            except:
                _precision = 1
            _recall = _tp / (_tp + _fn)
            _acc = (_tp + _tn)/len(label)

            self.threshold = np.append(self.threshold, (thres/100.))
            
            self.tpr = np.append(self.tpr, _tpr)
            self.fpr = np.append(self.fpr, _fpr)
            self.tnr = np.append(self.tnr, _tnr)
            
            self.tp = np.append(self.tp, _tp)
            self.tn = np.append(self.tn, _tn)
            self.fp = np.append(self.fp, _fp)
            self.fn = np.append(self.fn, _fn)
            
            self.prec = np.append(self.prec, _precision)
            self.recall = np.append(self.recall, _recall)
            self.acc = np.append(self.acc, _acc)
        self.tpr[0] = 1.0
        self.fpr[0] = 1.0		
	
    def calc_youden_J_statistic(self):
        _J = self.tpr + self.tnr - 1.
        _youden_J = _J.max()
        _youden_J_threshold = self.threshold[_J.argmax()] 
        
        return [_youden_J, _youden_J_threshold]
    
    def give_threshold(self, given_threshold=0.5):
        _arg = np.argwhere(self.threshold==given_threshold)
        
              
        _recall = np.asscalar(self.recall[_arg])
        _precision = np.asscalar(self.prec[_arg])
        _acc = np.asscalar(self.acc[_arg])
        _tp = np.asscalar(self.tp[_arg])
        _tn = np.asscalar(self.tn[_arg])
        _fp = np.asscalar(self.fp[_arg])
        _fn = np.asscalar(self.fn[_arg])
        _tpr = np.asscalar(self.tpr[_arg])
        _fpr = np.asscalar(self.fpr[_arg])
        _tnr = np.asscalar(self.tnr[_arg])
        
        return {'threshold':given_threshold,
                'recall':_recall ,
                'precision':_precision,
				'accuracy':_acc,
                'TP':_tp,
                'TN':_tn,
                'FP':_fp,
                'FN':_fn,
                'TPR':_tpr,
                'FPR':_fpr,
                'TNR':_tnr
        }
        
                

class MultiClassMetrics():

    def __init__(self, prob,pred,label):
        
        self.n_classes = len(prob.keys())
        self.n_labels = len(np.unique(label))
        self.n_pred = len(np.unique(pred))
        self.n_obs = len(pred)
        self.accuracy = len(pred[pred==label])/len(pred)
        
        
        count_uniques = np.bincount(pred)
        ii = np.nonzero(count_uniques)[0]
        self.class_counts = {}
        for i in range(0,len(count_uniques)):
            self.class_counts[str(ii[i])] = count_uniques[ii][i]
        
        
        if self.n_classes == self.n_labels == self.n_pred:
            print('GOOD TO GO')
        
        multi = {}
        for i, j in enumerate(prob.keys()):

            _label = label.copy()
            _label[label==i] = 1
            _label[label!=i] = 0
            
            _pred = pred.copy()
            _pred[pred==i] = 1
            _pred[pred!=i] = 0
            
            multi[j] = ClassMetrics(prob[j], _label)
        
        self.multi = multi
        self.micro_macro()
        
    def micro_macro(self):

        self.precision = {'macroA':np.empty(0),
                          'macroB':np.empty(0),
                          'micro':np.empty(0)
                         }
        
        self.recall    = {'macroA':np.empty(0),
                          'macroB':np.empty(0),
                          'micro':np.empty(0)
                         }
        
        self.tpr = {'macroA':np.empty(0),
                    'macroB':np.empty(0),
                    'micro':np.empty(0)
                    }
        
        self.fpr = {'macroA':np.empty(0),
                    'macroB':np.empty(0),
                    'micro':np.empty(0)
                    }
        
        self.threshold = np.empty(0)
        

        for j in range(0,len(self.multi[list(self.multi.keys())[0]].threshold)):
            
            _prec_macroA = 0
            _prec_macroB = 0
            _recall_macroA = 0
            _recall_macroB = 0
            _tpr_macroA = 0
            _tpr_macroB = 0
            _fpr_macroA = 0
            _fpr_macroB = 0
            
            _tp = 0
            _fp = 0
            _fn = 0
            _tn = 0
            
            for i in self.multi.keys():
                _prec_macroA += (self.multi[i].prec[j]/self.n_classes)
                _recall_macroA += (self.multi[i].recall[j]/self.n_classes)
                _tpr_macroA += (self.multi[i].tpr[j]/self.n_classes)
                _fpr_macroA += (self.multi[i].fpr[j]/self.n_classes)                
                
                _prec_macroB += (self.multi[i].prec[j]*(self.class_counts[i]/self.n_obs))
                _recall_macroB += (self.multi[i].recall[j]*(self.class_counts[i]/self.n_obs))          				
                _tpr_macroB += (self.multi[i].tpr[j]*(self.class_counts[i]/self.n_obs))   
                _fpr_macroB += (self.multi[i].fpr[j]*(self.class_counts[i]/self.n_obs))                   
                
                _tp += (self.multi[i].tp[j])
                _fp += (self.multi[i].fp[j])     
                _fn += (self.multi[i].fn[j])
                _tn += (self.multi[i].tn[j])
            
            self.threshold = np.append(self.threshold, self.multi['0'].threshold[j])
            
            self.precision['macroA'] = np.append(self.precision['macroA'], _prec_macroA)
            self.recall['macroA'] = np.append(self.recall['macroA'], _recall_macroA)    
            self.tpr['macroA'] = np.append(self.tpr['macroA'], _tpr_macroA)   
            self.fpr['macroA'] = np.append(self.fpr['macroA'], _fpr_macroA)               
            
            self.precision['macroB'] = np.append(self.precision['macroB'], _prec_macroB)
            self.recall['macroB'] = np.append(self.recall['macroB'], _recall_macroB)    
            self.tpr['macroB'] = np.append(self.tpr['macroB'], _tpr_macroB)   
            self.fpr['macroB'] = np.append(self.fpr['macroB'], _fpr_macroB)                   

            
            if (_tp != 0) and (_fp != 0):
                self.precision['micro']  = np.append(self.precision['micro'], _tp /(_tp + _fp))
            else:
                self.precision['micro'] = np.append(self.precision['micro'], 1.0)
            
            self.recall['micro']  = np.append(self.recall['micro'], _tp /(_tp + _fn))  
            self.fpr['micro'] = np.append(self.fpr['micro'], _fp/ (_fp + _tn))
            self.tpr['micro'] = np.append(self.tpr['micro'], _tp /(_tp + _fn))       

    
    def give_threshold(self, given_threshold=0.5, method='micro'):
        _arg = np.argwhere(self.threshold==given_threshold)
        
        
        _recall = np.asscalar(self.recall[method][_arg])
        _precision = np.asscalar(self.precision[method][_arg])
        _tpr = np.asscalar(self.tpr[method][_arg])
        _fpr = np.asscalar(self.fpr[method][_arg])
        
        
        _acc = self.accuracy
        
        return {'threshold':given_threshold,
                'recall':_recall ,
                'precision':_precision,
                'accuracy':_acc,
                'TPR':_tpr,
                'FPR':_fpr
        }
		
    def calc_youden_J_statistic(self, method='micro'):
        _J = self.tpr[method] + self.tnr[method] - 1.
        _youden_J = _J.max()
        _youden_J_threshold = self.threshold[_J.argmax()] 
        
        return [_youden_J, _youden_J_threshold]

def ftr_importance(cols, imps):
    d={}
    for i in range(0, len(cols)):
        d[cols[i]] = imps[i]*100.
    
    d_values = sorted(d.values()) 
    d_keys = sorted(d, key=d.get)
    d_keys.reverse()
    d_values.reverse()
    
    return d_keys, d_values

#def cluster_corr(corr): # No longer needed
#    
#    _corr = corr.abs()
#    
#    
#    _clustering = AffinityPropagation().fit(_corr)
#
#    _corr['cluster'] = _clustering.labels_
#    
#    _cols = list(_corr.columns)
#    _clus = list(_clustering.labels_)
       
#    order = {}
#    for i in range(0,len(_clus)): 
#        order[_cols[i]] = _clus[i]
        
#    x = OrderedDict(sorted(order.items(), key=lambda x: x[1]))
#    clus_corr = _corr.reindex(x.keys())[list(x.keys())]
    
#    return x, clus_corr
        
def corr_with_label(df, label, method='pearson'):

    """
    Computes clustered correlation matrix for a confusion matrix held in a pandas DataFrame.
	You can calculate a correlation matrix by calling .corr(). 
	You can calculate a correlation matrix for a spark dataframe by calling MLLytics.spark.spark_corr_matrix
    :param df: Pandas dataframe of all columns you wish to compare with the label
	:param label: Column to compare correlations with
	:param method: Correlation technique to pass to pd.corr()
    :return: Dictionary of column names and correlation values
    """

    _df = df.select_dtypes(include=[np.number])
    _x = {}
    for col in _df.columns:
        print(col, _df[col].corr(_df[label], method=method))
        _x[col] = (_df[col].corr(_df[label], method=method))
    
    return _x

        
def cluster_correlation_matrix(corr: pd.DataFrame):
    """
    Computes clustered correlation matrix for a confusion matrix held in a pandas DataFrame.
	You can calculate a correlation matrix by calling .corr(). 
	You can calculate a correlation matrix for a spark dataframe by calling MLLytics.spark.spark_corr_matrix
	
    :param df: Pandas dataframe of correlation matrix
    :return: Tuple containing dict of features with cluster assignment and correlation matrix ordered in the same way
    """
    # Compute correlation matrix and taks abs value
    #corr: pd.DataFrame = spark_corr_matrix(df)
    abs_corr = corr.abs()
    
    # Use AffinityPropagation to cluster data
    affinity_prop = AffinityPropagation().fit(abs_corr)
    abs_corr["cluster"] = affinity_prop.labels_
    
    # Extract columns and labels
    cols = list(abs_corr.columns)
    labels = list(affinity_prop.labels_)
    
    # Order the columns
    order = {cols[i]: labels[i] for i in range(len(labels))}
    order = OrderedDict(sorted(order.items(), key = lambda x: x[1]))
    
    # Reindex DataFrame
    keys = list(order.keys())
    clustered_corr = abs_corr.reindex(order.keys())[keys]
    
    return order, clustered_corr 
	

        
def cross_val(_df, label, model, k_folds=5):

    """
    _df is a pandas dataframe of your predictive and target variables (i.e. both X and y). This 
	function then splits that dataframe into k_folds, training/testing on a given model. It returns 
	a tuple of the best fit model as a ClassMetrics object, and a list for all individual folds of 
	ClassMetrics objects.
	
    :param _df: Pandas dataframe of all data (X and y)
	:param label: Label of column to predict
	:param model: Given scikit-learn model
	:param k_folds: Number of folds to cross validate on. 
    :return: Tuple containing ClassMetrics object of first the best fit model, and then a list of objects for each fold. 
    """


    cv_df = _df.sample(frac=1).copy()
    cv_df['row_num'] = np.arange(len(_df))
    
    fold_size = len(_df)/k_folds
    
    out = []    
    _acc = []
    _auc = []


    _prob_save = []
    _label_save = []
    
    for i in range(0,k_folds):
        vals = np.arange(i*fold_size,(i*fold_size)+fold_size)
        test_df = cv_df[cv_df['row_num'].isin(vals)]
        train_df = cv_df[~cv_df['row_num'].isin(vals)]
        
        X_train = train_df.loc[:, train_df.columns != label]
        X_test  = test_df.loc[:, test_df.columns != label]
        y_train = train_df[[label]]
        y_test = test_df[[label]]        

        model.fit(X_train, y_train[label].ravel())
        
        _pred_proba = model.predict_proba(X_test)
        _prob = np.array([_pred_proba[i][1] for i in range(0,len(_pred_proba))])
        _label = y_test[label].ravel()
        

        _cl = ClassMetrics(_prob, _label)

        _auc.append(auc(_cl.fpr,_cl.tpr))
        _acc.append(model.score(X_test, y_test))
        
        _prob_save.append([_prob])
        _label_save.append([_label])


        out.append(_cl)

    print('The 5 AUC scores were: ', _auc)
    print('The 5 ACC scores were: ', _acc)
    
    
    best_model = {'model':out[np.argmax(np.array(_auc))],
                  'prob':np.array(_prob_save[np.argmax(np.array(_auc))][0]),
                  'label':np.array(_label_save[np.argmax(np.array(_auc))][0])
                 }
    
    return best_model, out		