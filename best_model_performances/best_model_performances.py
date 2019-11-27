import pandas as pd
import numpy as np
from sys import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix, fbeta_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

data_dir = argv[1]
file_name = argv[2]

# setting parameters, also for the numer of folds
random_state = 0

# loading the training set based on the path given with the command line arguments
training_set = pd.read_csv('%straining.csv'%(data_dir), sep=' ')

# collecting all column names that are not the class
# consequently those are the feature columns
features = []
for col in training_set.columns:
	if col != 'class':
		features.append(col)

# create the feature set and class vector
X = np.array(training_set[features])
y = np.array(training_set['class'])

# Initialization of classification models with the settings that 
# achieved the best performances
# Gradient Boosting: 130 estimators, learning-rate=0.18
# Random Forest: 200 estimators, max-depth = 17
# Support Vector: kernel='poly', degree=2, coef0=2, gamma='scale'
clfs = []
clf1 = GradientBoostingClassifier(random_state=0, n_estimators=130, learning_rate=0.18)
clfs.append(('Gradient Boosting', clf1))
clf2 = RandomForestClassifier(random_state=0, max_depth=17, n_estimators=200) # approx: 0.75
clfs.append(('Random Forest', clf2))
clf3 = SVC(kernel='poly', degree=2, coef0=2, gamma='scale', probability=True)
clfs.append(('Support Vector', clf3))

# collect curve information
rocs = []
prcs = []

# load testing set
testing_set = pd.read_csv('%stesting.csv'%(data_dir), sep=' ')
final_test_X = testing_set[features]
final_test_y = testing_set['class']
balance_test = np.mean(final_test_y)

for name, clf in clfs:
	# train model on the training set
	model = clf.fit(X, y)
	probas = model.predict_proba(final_test_X)

	# calculating area under the curves (ROC, Precision-Recall)
	fpr, tpr, thresholds = roc_curve(final_test_y, probas[:, 1])
	roc_auc = auc(fpr, tpr)
	precision, recall, thresholds = precision_recall_curve(final_test_y, probas[:,1])
	prc_auc = auc(recall, precision)
	print('Performance on Test Set; Classifier:', name)
	print('auROC: %.3f'%(roc_auc))
	print('auPRC: %.3f\n'%(prc_auc))
	
	rocs.append((name, fpr, tpr, roc_auc))
	prcs.append((name, recall, precision, prc_auc))

# define settings for the plots
fs_title = 28
fs_ticks = 18
fs_labels = 22
fs_legend = 20
curve_ticks = [x/10 for x in range(0,12,2)]

# plot the ROC-curve
curve_width = 5
dashed_line_width = 2
fig = plt.figure(figsize=(10,10.3))
for name, fpr, tpr, auc in rocs:
	plt.plot(fpr, tpr, 
				linewidth=curve_width,
				label='%.2f | %s'%(auc, name))
plt.plot([0, 1], [0, 1], 'k--', lw=dashed_line_width)
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate', fontsize=fs_labels+4)
plt.ylabel('True Positive Rate', fontsize=fs_labels+4)
plt.xticks(fontsize=fs_ticks, ticks=curve_ticks)
plt.yticks(fontsize=fs_ticks, ticks=curve_ticks)
plt.title('ROC-curve', fontsize=fs_title, fontweight='bold')
plt.legend(loc="lower right", fontsize=fs_legend, title='Area under Curve', title_fontsize=fs_legend+2)
plt.tight_layout()
fig.savefig('./ROC-curve_%s.png'%(file_name))
plt.clf()


# plot the PR-curve
fig = plt.figure(figsize=(10,10.3))
for name, recall, precision, auc in prcs:
	plt.plot(recall, precision,  
			linewidth=curve_width,
			label='%.2f | %s'%(auc, name))
plt.plot([0, 1], [balance_test, balance_test], 'k--', lw=dashed_line_width)
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall', fontsize=fs_labels+4)
plt.ylabel('Precision', fontsize=fs_labels+4)
plt.xticks(fontsize=fs_ticks, ticks=curve_ticks)
plt.yticks(fontsize=fs_ticks, ticks=curve_ticks)
plt.title('PR-curve', fontsize=fs_title, fontweight='bold')
leg_pos = "upper right"
if 'everything' in data_dir:
	leg_pos = "lower right"
plt.legend(loc=leg_pos, fontsize=fs_legend, title='Area under Curve', title_fontsize=fs_legend+2)
plt.tight_layout()
fig.savefig('PR-curve_%s.png'%(file_name))
plt.clf()





























