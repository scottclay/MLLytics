# MLLytics

## Installation instructions
```pip install MLLytics```
or
```python setup.py install```
or
``` conda env create -f environment.yml```

## Update pypi instructions (for me)
Creates the package
```python setup.py sdist bdist_wheel```
Upload package
```twine upload --repository pypi *version_files*```

## Future
### Improvements and cleanup
* Comment all functions and classes
* Add type hinting to all functions and classes (https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html)
* Scoring functions
* More output stats in overviews
* Update reliability plot https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
* Tests
* Switch from my metrics to sklearn metrics where it makes sense? aka
```fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])```
and more general macro/micro average metrics from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
* Additional metrics (sensitivity, specificity, precision, negative predictive value, FPR, FNR,
false discovery rate, accuracy, F1 score

### Cosmetic
* Fix size of confusion matrix
* Check works with matplotlib 3
* Tidy up legends and annotation text on plots
* Joy plots
* Brier score for calibration plot
* Tidy up cross validation and plots (also repeated cross-validation)
* Acc-thresholds graph

### Recently completed
* ~Allow figure size and font sizes to be passed into plotting functions~
* ~Example guides for each function in jupyter notebooks~
* ~MultiClassMetrics class to inherit from ClassMetrics and share common functions~
* ~REGRESSION~

## Contributing Authors
* Scott Clay
* David Sullivan
