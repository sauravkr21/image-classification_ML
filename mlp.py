import numpy as np
import pickle

# Creating a MLP class with 2 hidden layers and 1 output layer


class MLP:
    # Initializing the weights and biases
    def __init__(self):
        self.layer1 = np.random.uniform(-1, 1, (64, 512))
        self.bias1 = np.random.uniform(-1, 1, (64))
        self.layer2 = np.random.uniform(-1, 1, (64, 64))
        self.bias2 = np.random.uniform(-1, 1, (64))
        self.layer3 = np.random.uniform(-1, 1, (10, 64))
        self.bias3 = np.random.uniform(-1, 1, (10))

    # Activation functions

    def softmax(self, x):
        exps = np.exp(x)
        # in case of all zeros
        if np.sum(exps) == 0:
            return exps
        return exps / np.sum(exps)

    def relu(self, x):
        return np.maximum(0, x)

    # Forward propagation

    def forward(self, x):
        self.a1 = self.relu(np.dot(self.layer1, x) + self.bias1)
        self.a2 = self.relu(np.dot(self.layer2, self.a1) + self.bias2)
        # divide by max magnitude to avoid overflow
        self.i = np.dot(self.layer3, self.a2) + self.bias3
        self.j = self.i / np.max(np.abs(self.i))
        self.a3 = self.softmax(self.j)
        return self.a3

    # One hot encoding
    def one_hot(self, x):
        y = np.zeros(10)
        y[x] = 1
        return y

    # Loss function
    def loss(self, y, pred):
        return -np.sum(y * np.log(pred+1e-10))

    # Backward propagation

    def backward(self, x, y, pred):
        # calculate error
        self.dz3 = pred - y
        # divide by max magnitude to avoid overflow
        self.dz3 /= np.max(np.abs(self.i))
        # calculate gradient for output layer
        self.dw3 = np.outer(self.dz3, self.a2)
        self.db3 = self.dz3
        # calculate gradient for hidden layer 2
        self.dz2 = np.dot(self.layer3.T, self.dz3) * \
            (self.a2 > 0).astype(np.float64)
        self.dw2 = np.outer(self.dz2, self.a1)
        self.db2 = self.dz2
        # calculate gradient for hidden layer 1
        self.dz1 = np.dot(self.layer2.T, self.dz2) * \
            (self.a1 > 0).astype(np.float64)
        self.dw1 = np.outer(self.dz1, x)
        self.db1 = self.dz1

    # Updating the weights and biases
    def update(self, lr):
        self.layer3 -= lr * self.dw3
        self.bias3 -= lr * self.db3
        self.layer2 -= lr * self.dw2
        self.bias2 -= lr * self.db2
        self.layer1 -= lr * self.dw1
        self.bias1 -= lr * self.db1

    # Training the model by SGD with RMSProp
    def train(self, x, lbl, lr):
        pred = self.forward(x)
        y = self.one_hot(lbl)
        self.backward(x, y, pred)
        self.update(lr)

    # Predicting the label
    def predict(self, x):
        pred = self.forward(x)
        return np.argmax(pred)


# Creating 2 MLP classifiers
mlp1 = MLP()
mlp2 = MLP()

# load the faeture vector and labels
with open('features.pkl', 'rb') as f:
    fvec = pickle.load(f)
with open('augmented_lbl.pkl', 'rb') as f:
    lbl = pickle.load(f)

# Normalizing the data
mean = np.mean(fvec)
std = np.std(fvec)
fvec = (fvec - mean) / std


# Training the model
lr = 0.03
# Training the model with 5000 samples and adaptive learning rate I have trained once u can do it again if you want
for i in range(100000):
    pred = mlp1.predict(fvec[i])
    c = 0
    while pred != lbl[i]:
        mlp1.train(fvec[i], lbl[i], lr)
        c += 1
        if c > 100:
            break
        pred = mlp1.predict(fvec[i])
    if i % 10000 == 0:
        print("Epoch: ", i, "Predicted label: ", pred, "Actual label: ",
              lbl[i], "Loss: ", mlp1.loss(mlp1.one_hot(lbl[i]), mlp1.forward(fvec[i])))

for i in range(50000):
    pred = mlp2.predict(fvec[i])
    c = 0
    while pred != lbl[i]:
        mlp2.train(fvec[i], lbl[i], lr)
        c += 1
        if c > 100:
            break
        pred = mlp2.predict(fvec[i])
    if i % 10000 == 0:
        print("Epoch: ", i, "Predicted label: ", pred, "Actual label: ",
              lbl[i], "Loss: ", mlp1.loss(mlp1.one_hot(lbl[i]), mlp1.forward(fvec[i])))

# load the test data
with open('test_vectors.pkl', 'rb') as f:
    test_fvec = pickle.load(f)

# Normalizing the test data
test_fvec = (test_fvec - mean) / std


# Predicting the labels
pred1 = []
pred2 = []
for i in range(10000):
    pred1.append(mlp1.predict(test_fvec[i]))
    pred2.append(mlp2.predict(test_fvec[i]))
pred1 = np.array(pred1)
pred2 = np.array(pred2)

# save the predictions
with open('pred1.pkl', 'wb') as f:
    pickle.dump(pred1, f)

with open('pred2.pkl', 'wb') as f:
    pickle.dump(pred2, f)

# save the models
with open('model1.pkl', 'wb') as f:
    pickle.dump(mlp1, f)
with open('model2.pkl', 'wb') as f:
    pickle.dump(mlp2, f)
