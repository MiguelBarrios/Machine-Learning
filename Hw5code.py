from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np
import csv

def list_to_csv(a, file_name):
	with open(file_name, "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(a)

def split_data(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices =shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

def TSS(y_true, y_pred):
	return np.sum((y_pred - np.average(y_true))**2)

def RSS(y_true, y_pred):
	return np.sum((y_true - y_pred)**2)

def R_squared(y_true, y_pred):
	rss = RSS(y_true, y_pred)
	tss = TSS(y_true, y_pred)
	return 1 - (rss/tss)

out = []
out.append(['RRP','RLP','MSE_TRAIN', 'MSE_VAL'])

df = pd.read_csv("superconduct/train.csv")

# Exercise 1.a   
train_set, test_set = split_data(df, 0.2)

# Exercise 1.b
validation, training = split_data(train_set, 0.75)

attributes = training.columns[:-1]
target = 'critical_temp'
x_training = training[attributes].values
y_training = training[target].values
x_validation = validation[attributes].values
y_validation = validation[target].values

penalties = [0,0.000001,0.00001,0.0001,0.001,0.01,0.1, 1]

for l1 in tqdm(penalties):
	for l2 in penalties:
		if l1 == 0 and l2 == 0:
			model = LinearRegression()
		else: 
			alpha = l1 + l2
			l1_ratio = l1 / alpha
			model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 50000)
		### Exercise 2.A ###
		model.fit(x_training,y_training)
		### Exercise 2.B ###
		y_train_pred = model.predict(x_training)
		### Exercise 2.C ###
		y_validation_pred = model.predict(x_validation)
		### Exercise 2.D ###
		ms1 = mean_squared_error(y_training, y_train_pred, squared = True)
		ms2 = mean_squared_error(y_validation, y_validation_pred, squared = True) 
		out.append([l1,l2,ms1, ms2])
list_to_csv(out, "output.csv")


# Excercise 2.f
l1 = 0.000001
l2 = 0.00001
alpha = l1 + l2
l1_ratio = l1 / alpha
x_train = train_set[attributes]
y_train = train_set[target]
x_test = test_set[attributes]
y_test = test_set[target]

model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# Excercise 3.a
out = R_squared(y_test.values, y_pred)
print(out)





