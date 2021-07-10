from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def split_data(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices =shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

def calc_cros_entropy(m,p, y):
	epsilon = 1e-5    # fix to log problem
	out = 0
	for i in range(m):
		out += y[i] * np.log(p[i] + epsilon) + (1 - y[i]) * np.log(1 - p[i] + epsilon)
	return (-1 /m * out)

model_prob_results = []

######################## Exercise 1: Preprocessing #####################
filepath = "Data_for_UCI_named.csv"
df = pd.read_csv(filepath)

# A: Remove Columns 5 and 13
df = df.drop(['p1', 'stab'], axis = 1)

# B: Change the target variable to a number
df['stabf'] = df['stabf'].replace(['stable'], 1)
df['stabf'] = df['stabf'].replace(['unstable'], 0)

# C: Split data into train(80%) and test set(20%)
train_val_set, test_set = split_data(df, 0.2)

# D: Split data into training 75% validation 25%
validation, train_set = split_data(train_val_set, 0.75)

X_train_val = train_val_set.iloc[:,0:11]
y_train_val = train_val_set['stabf'].values

X_test = test_set.iloc[:,0:11]
y_test = test_set['stabf'].values

X_train = train_set.iloc[:,0:11]
y_train = train_set['stabf'].values

X_val = validation.iloc[:,0:11]
y_val = validation['stabf'].values

######################## Exercise 2: Decision Trees #####################

# train a new decision tree on the training and validation data using 
# whichever measure created the best model in homework 4, with a max tree depth of 5.
# make a probabilistic prediction for each testing example
tree_clif = DecisionTreeClassifier(max_depth = 5, criterion = "gini")
tree_clif.fit(X_train_val,y_train_val)

prob_tree = tree_clif.predict_proba(X_test)

######################## Exercise 3: Neural Network #####################

# A: Fit a neural network to the training data using 1 hidden layer of 20 units
# 	 as well as another neural network that has 2 hidden layers of 10 units each.
nn_clif_1 = MLPClassifier(hidden_layer_sizes = (20,), max_iter = 1000)
nn_clif_2 = MLPClassifier(hidden_layer_sizes = (10,10,), max_iter = 1000)

nn_clif_1.fit(X_train,y_train)
nn_clif_2.fit(X_train,y_train)

# B: For each model made in (a), make a probabilistic prediction for each validation 
#    example. Re- port the cross-entropies between the predictions and the true labels
#    in your writeup.

prob_nn_clif_1 = nn_clif_1.predict_proba(X_val)
prob_nn_clif_2 = nn_clif_2.predict_proba(X_val)

cross_entropy_nn_1 = calc_cros_entropy(len(y_val),prob_nn_clif_1, y_val)
cross_entropy_nn_2 = calc_cros_entropy(len(y_val),prob_nn_clif_2, y_val)

# results max iter 1000
#cross_entropy_nn_1 = array([5.88660115, 0.14814241])
# cross_entropy_nn_2 = array([5.99446839, 0.15498536])

# C: Which neural network performs the best on the validation data? Report this in your 
#    writeup. Train a new neural network using the architecture that performed better among 
#    the two using the training and validation data. 

# The neural network trained with 1 hidden layer and 20 nodes, nn1

nn_clif = MLPClassifier(hidden_layer_sizes = (20,), max_iter = 1000)
nn_clif.fit(X_train_val,y_train_val)

# Make a probabilistic prediction for each testing example using this model 
# and save them for later.

prob_nn = nn_clif.predict_proba(X_test)
model_prob_results.append(prob_nn)

######################## Exercise 4: Random Forest #####################

"""
A:  Fit random forests to the training data using 5, 10, 50, 100, and 200 trees 
	in each forest, using 4 randomly-selected features at each split and a max 
	tree depth of 5. Use the measure that performed the best on the data in problem 2.
"""
rf5 = RandomForestClassifier(n_estimators=5,max_features=4,criterion='gini', max_depth=5)
rf10 = RandomForestClassifier(n_estimators=10,max_features=4,criterion='gini', max_depth=5)
rf50 = RandomForestClassifier(n_estimators=50,max_features=4,criterion='gini', max_depth=5)
rf100 = RandomForestClassifier(n_estimators=100,max_features=4,criterion='gini', max_depth=5)
rf200 = RandomForestClassifier(n_estimators=200,max_features=4,criterion='gini', max_depth=5)

rf5.fit(X_train,y_train)
rf10.fit(X_train,y_train)
rf50.fit(X_train,y_train)
rf100.fit(X_train,y_train)
rf200.fit(X_train,y_train)

"""
B: For each model made in (a), make a probabilistic prediction for each validation example.
   Report the cross-entropies between the predictions and the true labels in your writeup.
"""

prob_pred_rf5 = rf5.predict_proba(X_val)
prob_pred_rf10= rf10.predict_proba(X_val)
prob_pred_rf50 = rf50.predict_proba(X_val)
prob_pred_rf100 = rf100.predict_proba(X_val)
prob_pred_rf200 = rf200.predict_proba(X_val)
cross_entropy_rf5 = calc_cros_entropy(len(y_val),prob_pred_rf5, y_val)
cross_entropy_rf10 = calc_cros_entropy(len(y_val),prob_pred_rf10, y_val)
cross_entropy_rf50 = calc_cros_entropy(len(y_val),prob_pred_rf50, y_val)
cross_entropy_rf100 = calc_cros_entropy(len(y_val),prob_pred_rf100, y_val)
cross_entropy_rf200 = calc_cros_entropy(len(y_val),prob_pred_rf200, y_val)

""" cross entropy results
cross_entropy_rf5 = array([1.47167956, 0.40716877])
cross_entropy_rf10 = array([1.43478096, 0.39335266])
cross_entropy_rf50 = array([1.38214344, 0.39120359])
cross_entropy_rf100 = array([1.38217905, 0.38938223])
cross_entropy_rf200 = array([1.37292828, 0.39234317])
"""


"""
C: Which number of trees performs the best on the validation data? Report this in your
   writeup. Train a new random forest using this number of trees using the training and
   validation data. Make a probabilistic prediction for each testing example using this
   model and save them for later.

   the tree with 200 performed best
"""

rf200 = RandomForestClassifier(n_estimators=200,max_features=4,criterion='gini', max_depth=5)
rf200.fit(X_train_val, y_train_val)

prob_pred_rf200_2 = rf200.predict_proba(X_test)
model_prob_results.append(prob_pred_rf200_2)

######################## Exercise 5: Boosting #####################
"""
 A: Fitboosteddecisionstumps(maxtreedepthof1)to the training data allowing at most 20 , 
 40, and 60 decision stumps (base estimators) in each model.
 """
ada_20 = AdaBoostClassifier(base_estimator = None, n_estimators = 20)
ada_40 = AdaBoostClassifier(base_estimator = None, n_estimators = 40)
ada_60 = AdaBoostClassifier(base_estimator = None, n_estimators = 60)

ada_20.fit(X_train,y_train)
ada_40.fit(X_train,y_train)
ada_60.fit(X_train,y_train)

"""
B: For each model trained in (a), make a probabilisitc prediction for each validation example.
   Report the cross-entropies between the predcitions and the true labels in your writeup.
"""
prob_pred_ada20 = ada_20.predict_proba(X_val)
prob_pred_ada40 = ada_40.predict_proba(X_val)
prob_pred_ada60 = ada_60.predict_proba(X_val)

cross_entropy_ada20 = calc_cros_entropy(len(y_val),prob_pred_ada20, y_val)
cross_entropy_ada40 = calc_cros_entropy(len(y_val),prob_pred_ada40, y_val)
cross_entropy_ada60 = calc_cros_entropy(len(y_val),prob_pred_ada60, y_val)


""" Cross entropy results
s
"""

"""
C: Which upper bound on the number of allowed base classifiers generates the best performing 
model? Report this in your writeup. Train a new AdaBoost classifier using this bound on 
the number of maximum allowed base classifiers, using the training and validation data. 
Make a probabilistic prediction for each testing example using this model and save them 
for later.

"""
ada_60.fit(X_train_val,y_train_val)
prob_pred_ada60 = ada_60.predict_proba(X_test)
model_prob_results.append(prob_pred_ada60)
######################## Exercise 6: ROC Curve #####################


def determinize(prob_threshold, probabilities):
	determinized_arr = np.zeros(len(probabilities))
	for i in range(len(probabilities)):
		determinized_arr[i] = 1 if (probabilities[i][1] >= prob_threshold) else 0
	return determinized_arr


# TRUE POS, TRUE NEG, FALSE POS, FALSE NEG
def confusion_matrix(predictions, actual):
	TP, TN, FP, FN = 0,0,0,0
	for i in range(len(actual)):
		pred = predictions[i]
		actual_value = actual[i]
		# positive prediction "1"
		if pred == 1:
			if actual_value == 1:
				TP += 1
			else:
				FP += 1
		else: # negative prediction "0"
			if actual_value == 0:
				TN += 1
			else:
				FN += 1
	return TP, TN, FP, FN

probabilities = np.zeros(1001)
prob = 0.0
index = 0
while prob <= 1:
	probabilities[index] = prob
	prob += 0.001
	prob = round(prob,3)
	index += 1

TPRS = []
FPRS = []
for prob_results in model_prob_results:
	cur_model_tpr_list = []
	cur_model_fpr_list = []
	for threshold in probabilities:
		# A
		det = determinize(threshold, prob_results)
		# B
		TP, TN, FP, FN = confusion_matrix(det, y_test)
		TNR = TN / (TN + FP)
		FPR = 1 - TNR
		TPR = TP / (TP + FN)
		cur_model_tpr_list.append(TPR)
		cur_model_fpr_list.append(FPR)
	TPRS.append(cur_model_tpr_list)
	FPRS.append(cur_model_fpr_list)

plt.plot(FPRS[0], TPRS[0], label = 'NN')
plt.plot(FPRS[1], TPRS[1], label = 'Random Forest')
plt.plot(FPRS[1], TPRS[2], label = 'Ada60')
plt.annotate("(0,0)", (0,0))
plt.annotate("(1,1)", (1,1))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# Find the probability threshold yielding the highest Youden index (TPR - FPR). Report the Youden index and the corresponding probability threshold for each model.

def Youden_prob_thresh_max(TPR, FPR):
	tpr = np.asarray(TPR)
	fpr = np.asarray(FPR)
	res = tpr - fpr
	max_index = np.where(res == np.amax(res))
	return max_index

nn_index = Youden_prob_thresh_max(TPRS[0],FPRS[0])
rf_index = Youden_prob_thresh_max(TPRS[1],FPRS[1])
ada60_index = Youden_prob_thresh_max(TPRS[2],FPRS[2])


print("neural_network probability threshold: " + str(probabilities[nn_index[0]]) + " "  + str(probabilities[nn_index[-1]]))
# neural_network probability threshold: [0.302 0.303 0.304 0.305 0.306]
print("random forest probability threshold: " + str(probabilities[rf_index[0]]))
# random forest probability threshold: [0.397]
print("ada60 probability threshold: " + str(probabilities[ada60_index[0]]))
# ada60 probability threshold: [0.496]


# Compute the AUC (area under the curve). 
# Neural net AUC
y_scores = determinize(0.302, model_prob_results[0])
roc_auc_score(y_test, y_scores)

# Random Forest net AUC
y_scores = determinize(0.397, model_prob_results[1])
roc_auc_score(y_test, y_scores)

# Boosing ada60 AUC
y_scores = determinize(0.496, model_prob_results[2])
roc_auc_score(y_test, y_scores)




