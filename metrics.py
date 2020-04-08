"""
	Metrics used to evaluate performance.

	Dimity Miller, 2020
"""
import numpy as np
import sklearn.metrics

def accuracy(x, gt):
	predicted = np.argmin(x, axis = 1)
	total = len(gt)
	acc = np.sum(predicted == gt)/total
	return acc


def auroc(inData, outData, in_low = True):
	inDataMin = np.min(inData, 1)
	outDataMin = np.min(outData, 1)
	
	allData = np.concatenate((inDataMin, outDataMin))
	labels = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = in_low)

	return sklearn.metrics.auc(fpr, tpr)
