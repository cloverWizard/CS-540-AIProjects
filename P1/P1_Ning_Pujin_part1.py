import numpy as np 
import matplotlib.pyplot as plt

def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)

x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

# train data dimension 784*60000
# print('train data dimension: ' + str(len(x_train[0])) + ' ' + str(len(x_train)))
# test data dimension 784*10000
# print('test data dimension: ' + str(len(x_test[0])) + ' ' + str(len(x_test)))
# train output 60000
# print('train data dimension: ' + str(len(y_train)))
# test output 10000
# print('test data dimension: ' + str(len(y_test)))
print('data loading done')

f1 = open("p1q1.txt", "w+")
f2 = open("p1q2.txt", "w+")
f3 = open("p1q3.txt", "w+")
f4 = open("p1q4.txt", "w+")

f1.write(str(x_train[1][0]))
for i in range(len(x_train[1])-1):
    f1.write(",");
    f1.write(str(x_train[1][i]))
f1.close()

test_labels = [1,3]
# [0] is the way to get the index of the array
indices = np.where(np.isin(y_train,test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

# filter relevant data
x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]

# mark them with 0 or 1
y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1
y_t[y_t==test_labels[0]] = 0
y_t[y_t == test_labels[1]] = 1



# Todo: you may need to change some hyper-paramter like num_epochs and alpha, etc
num_epochs = 100
m = x.shape[1]
n = x.shape[0]
alpha = 0.3

large_num = 1e8
epsilon = 1e-6
thresh = 1e-4

w = np.random.rand(m)
b = np.random.rand()

c = np.zeros(num_epochs)


for epoch in range(num_epochs):
    # a = g(w^T+b)
    a = 1 / (1 + np.exp(-(np.matmul(w, np.transpose(x)) + b)))
    
    w -= alpha * np.matmul(a-y, x)
    
    b -= alpha * (a-y).sum()
    
    cost = np.zeros(len(y))
    idx = (y==0) & (a > 1 - thresh) | (y == 1) & (a < thresh)
    cost[idx] = large_num
    
    a[a<thresh] = thresh
    a[a> 1-thresh] = thresh
    
    inv_idx = np.invert(idx)
    cost[inv_idx] = - y[inv_idx] * np.log(a[inv_idx]) - (1-y[inv_idx]) * np.log(1-a[inv_idx])
    c[epoch] = cost.sum()
    
    if epoch % 3 == 0:
        print('epoch = ', epoch + 1, 'cost = ', c[epoch])
    
    if epoch > 0 and abs(c[epoch -1] - c[epoch]) < epsilon:
        break
    
f2.write(str(round(w[0], 4)))
for i in range(len(w)-1):
    f2.write(",")
    f2.write(str(round(w[i], 4)))
f2.write(",")
f2.write(str(round(b,4)))
f2.close()

# Todo: new test
new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0

a_new = 1 / (1 + np.exp(-(np.matmul(w, np.transpose(new_x)) + b)))

f3.write(str(a_new[0]))
f4.write(str(round(a_new[0])))
for i in range(len(a_new)-1):
    f3.write(",")
    f3.write(str(a_new[i]))
    f4.write(",")
    f4.write(str(round(a_new[i])))
f3.close()
f4.close()
# print(str(len(new_x)))



    