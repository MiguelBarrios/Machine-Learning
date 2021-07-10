from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

######################## Exercise 2: Gradiant Decent #####################

def h(x, w0, w1):
	return w0 + w1 * x

def Rs(training_examples, w0,w1):
	res = 0
	for training_example in training_examples:
		x = training_example[0]
		y = training_example[1]
		res += (y - h(x,w0,w1))**2
	return res / 2

def gradiant_decent(training_examples,alpha, num_itr):
	w0 = 1
	w1 = 1
	for itr in range(num_itr):
		delta0 = 0
		delta1 = 0
		for training_example in training_examples:
			x = training_example[0]
			y = training_example[1]
			delta0 += (y - (w0 + w1 * x))
			delta1 += (y - (w0 + w1 * x)) * x
		print("delta0 = {delta0} delta1 = {delta1}".format(delta0 = delta0, delta1 = delta1))
		w0 = w0 + alpha * delta0
		w1 = w1 + alpha * delta1
		print("w0 = {w0} w1 = {w1} Rss = {rs}\n".format(w0 = w0, w1 = w1, rs = Rs(training_examples, w0, w1)))

training_examples = [[1.2,3.2],[2.8,8.5],[2,4.7], [0.9,2.9],[5.1,11]]
gradiant_decent(training_examples, 0.01, 3)

######################## Exercise Preprocessing #####################

def split_data(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices =shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

filepath = "/Users/miguelbarrios/Documents/School/Machine Learning/HW/HW4/Data_for_UCI_named.csv"
df = pd.read_csv(filepath)

# (a) Remove columns 5 and 13
df = df.drop(['p1', 'stab'], axis = 1)


# (b) Change the target variable to a number. If the value is stable, change it to 1, and if the value is unstable, change it to 0.
df['stabf'] = df['stabf'].replace(['stable'], 1)
df['stabf'] = df['stabf'].replace(['unstable'], 0)

# (c) split data into train(80%) and test set(20%)
train_set, test_set = split_data(df, 0.2)

# (d) split data into training 75% validation 25%
validation, train_set_2 = split_data(train_set, 0.75)

######################## Exercise 4 #####################
def calc_cros_entropy(m,p, y):
	epsilon = 1e-5    # fix to log problem
	out = 0
	for i in range(m):
		out += y[i] * np.log(p[i] + epsilon) + (1 - y[i]) * np.log(1 - p[i] + epsilon)
	return (-1 /m * out)

#(a)
X_train = train_set_2.iloc[:,0:11]
y_train = train_set_2['stabf'].values

# (a) Fit a decision tree to the training data using GINI index and max tree depth of 5.
tree_clif = DecisionTreeClassifier(max_depth = 5, criterion = "gini")
tree_clif.fit(X_train,y_train)

# (b) Using the model created in part a make a probablistic prediction for each validation example
X = validation.iloc[:,0:11]
y = validation['stabf'].values
prob = tree_clif.predict_proba(X)
cross_entropy = calc_cros_entropy(len(y),prob, y)
print("cross_entropy GINI = " , end = "")
print(cross_entropy)


# (c) Fit a decision tree to the training data using information gain and max tree depth of 5
tree_clif = DecisionTreeClassifier(max_depth = 5, criterion = "entropy")
tree_clif.fit(X_train,y_train)

 # (d) Using the model created in part (c) make a probablistic prediction for each val example
prob = tree_clif.predict_proba(X)
cross_entropy = calc_cros_entropy(len(y),prob, y)
print("cross_entropy info gain= " , end = "")
print(cross_entropy)


# (e) Fit a decision tree to the training data using GINI index and max tree depth of 5
X_train = train_set.iloc[:,0:11]
y_train = train_set['stabf'].values
tree_clif = DecisionTreeClassifier(max_depth = 5, criterion = "gini")
tree_clif.fit(X_train,y_train)
X = test_set.iloc[:,0:11]
y = test_set['stabf'].values
prob = tree_clif.predict_proba(X)
cross_entropy = calc_cros_entropy(len(y),prob, y)
print("cross_entropy GINI F = " , end = "")
print(cross_entropy)
