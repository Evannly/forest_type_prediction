# Use GLUON to do Linear Classification
# (Chinese) http://zh.gluon.ai/chapter_supervised-learning/linear-regression-gluon.html
# (English) http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html

#%% LIBRARIES
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

#%% Generate dadaset
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# Read data
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

'''
# Print data
for data, label in data_iter:
    print(data, label)
    break
'''
#%% Create model
## Structure
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize()        # must init FIRST!

## Loss Function
square_loss = gluon.loss.L2Loss()
## Optimization Method
trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.1})


#%% Traianing
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

#%% Show result
dense = net[0]
true_w, dense.weight.data()
true_b, dense.bias.data()
