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
