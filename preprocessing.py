from sklearn.tree import DecisionTreeClassifier
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


# (e)
X_train = train_set.iloc[:,0:11]
y_train = train_set['stabf'].values

# (a) Fit a decision tree to the training data using GINI index and max tree depth of 5

tree_clif = DecisionTreeClassifier(max_depth = 5, criterion = "gini")
tree_clif.fit(X_train,y_train)
X = test_set.iloc[:,0:11]
y = test_set['stabf'].values
prob = tree_clif.predict_proba(X)
cross_entropy = calc_cros_entropy(len(y),prob, y)





print("cross_entropy GINI F = " , end = "")
print(cross_entropy)




























