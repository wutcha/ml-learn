import numpy as np

class MNIST:
    def __init__(self, epoch = 10, rate = 0.01, batch_size = 100):
        self.epoch = epoch
        self.rate = rate
        self.batch_size = batch_size

        self.rng = np.random.default_rng(67)

        self.w1 = self.rng.uniform(-0.02, 0.02, (784, 128))
        self.b1 = np.zeros(128)

        self.w2 = self.rng.uniform(-0.02, 0.02, (128, 64))
        self.b2 = np.zeros(64)

        self.w3 = self.rng.uniform(-0.02, 0.02, (64, 10))
        self.b3 = np.zeros(10)
    
    # need relu, forwardpass, loss calc + softmax, backprop
    
    def relu(self, x): 
        return np.maximum(0, x)
    
    def softmax(self, x):
        x = x-np.max(x,axis=1)[:,None]
        #print(x.shape)
        exp = np.exp(x)
        return (exp/np.sum(exp,axis=1)[:,None])

    def fit(self, data, labels):
        data = data/255.0

        for e in range(self.epoch):
            for b in range(0,data.shape[0],self.batch_size):

                z1 = data[b:b + self.batch_size] @ self.w1 + self.b1
                a1 = self.relu(z1)
                z2 = a1 @ self.w2 + self.b2
                a2 = self.relu(z2)
                z3 = a2 @ self.w3 + self.b3
                a3 = self.softmax(z3)
                
                #print(a1.shape, a2.shape, a3.shape)
                c_labels = labels[b:b + self.batch_size]
                #print(c_labels.shape)
                probs = a3[np.arange(self.batch_size), c_labels]
                #print(probs)
                #loss = -1*np.mean(training_risk * np.log(y) + (1 - training_risk)*np.log(1-y))
                loss = -1*np.mean(np.log(probs))

                # dL/dw3 = dL/da3 * da3/dz3 * dz3/dw3 => a
                # dL/dw2 = dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/dw2
                # dL/dw1 = dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
                # dz3 = a3-y
                
                y = np.zeros_like(a3)
                y[np.arange(len(c_labels)), c_labels] = 1

                dz3 = (a3 - y) / self.batch_size
                dw3 = a2.T @ (dz3)
                #print(dw3.shape)
                db3 = np.sum(dz3, axis = 0)

                dz2 = (dz3 @ self.w3.T) * (z2>0)
                dw2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0)
                #print(dz2.shape,dw2.shape,db2.shape)

                dz1 = (dz2@self.w2.T)*(z1>0)
                dw1 = data[b:b + self.batch_size].T @ dz1
                db1 = np.sum(dz1, axis=0)
                #print(dz1.shape,dw1.shape,db1.shape)

                self.w3 -= self.rate*dw3
                self.w2 -= self.rate*dw2
                self.w1 -= self.rate*dw1
                self.b3 -= self.rate*db3
                self.b2 -= self.rate*db2
                self.b1 -= self.rate*db1
            print(f"Epoch: {e+1} | Loss: {loss}")

    def predict(self, image):
        image = image/255.0
        a1 = self.relu(image @ self.w1 + self.b1)
        a2 = self.relu(a1 @ self.w2 + self.b2)
        a3 = self.softmax(a2 @ self.w3 + self.b3)

        return np.argmax(a3, axis=1)
         

with open("datasets/mnist/train-images.idx3-ubyte", "rb") as f:
    train_data = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(60000,784)

with open("datasets/mnist/train-labels.idx1-ubyte", "rb") as f:
    train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

with open("datasets/mnist/t10k-images.idx3-ubyte", "rb") as f:
    test_data = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(10000,784)

with open("datasets/mnist/t10k-labels.idx1-ubyte", "rb") as f:
    test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

if __name__ == '__main__':
    np.set_printoptions(linewidth=100)
    model = MNIST(epoch=50,rate=0.025,batch_size=100)
    model.fit(train_data, train_labels)
    print(model.w1.shape)
    x = model.predict(test_data)
    print(np.count_nonzero(x==test_labels)/10000)

    '''
    lr=0.01 | epoch=10 | bsize=100 | accuracy=86.4
    lr=0.01 | epoch=20 | bsize=100 | accuracy=91.27
    lr=0.02 | epoch=40 | bsize=100 | accuracy=96.8
    lr=0.025 | epoch=50 | bsize=100 | accuracy=97.46
                    
    
    
    '''
    