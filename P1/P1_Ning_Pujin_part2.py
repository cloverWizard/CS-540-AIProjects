import numpy as np 
import matplotlib.pyplot as plt


# Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

test_labels = [7,4]
indices = np.where(np.isin(y_train,test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]


test_labels = [1,3]
indices = np.where(np.isin(y_train,test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]

y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1
y_t[y_t==test_labels[0]] = 0
y_t[y_t == test_labels[1]] = 1
num_hidden_uints = 392


def relu(x):
    y = x
    y[y<0] = 0
    return y

def diff_relu(x):
    y = x
    y[x>0] = 1
    y[x<=0] = 0
    return y

def logis(x):
    return 1 / (1 + np.exp(-x))

def nnet(train_x, train_y, test_x, test_y, alpha, num_epochs):
    num_train = len(train_y)
    num_test = len(test_y)
    
    train_x = np.hstack((train_x, np.ones(num_train).reshape(-1,1)))
    test_x = np.hstack((test_x, np.ones(num_test).reshape(-1,1)))
    
    num_input_uints = train_x.shape[1]  # 785 
    
    
    wih = np.random.uniform(low=-1, high=1, size=(num_hidden_uints, num_input_uints)) #392*785
    who = np.random.uniform(low=-1, high=1, size=(1, num_hidden_uints+1)) # 1 * 393
    
    for epoch in range(1, num_epochs+1):
        out_o = np.zeros(num_train)
        out_h = np.zeros((num_train, num_hidden_uints+1))  #num_train * 393
        out_h[:,-1] = 1
        for ind in range(num_train):
            row = train_x[ind]  # len = 785 
            out_h[ind, :-1] = logis(np.matmul(wih, row))
            out_o[ind] = 1 / (1 + np.exp(-sum(out_h[ind] @ who.T)))

            delta = np.multiply(logis(out_h[ind]), (train_y[ind] - out_o[ind]) * np.squeeze(who))
            wih += alpha * np.matmul(np.expand_dims(delta[:-1], axis=1), np.expand_dims(row,axis=0))
            who += np.expand_dims(alpha * (train_y[ind] - out_o[ind]) * out_h[ind,:], axis=0)
        error = sum(- train_y * np.log(out_o) - (1-train_y) * np.log(1-out_o))
        num_correct = sum((out_o > 0.5).astype(int) == train_y)
        print('epoch = ', epoch, "error = ", str(error), 'correctly classified = {:.4%}'.format(num_correct / num_train))
        # print('epoch = ', epoch, ' error = {:.7}'.format(error), 'correctly classified = {:.4%}'.format(num_correct / num_train))
    
    return wih.T, who




# Todo: change these hyper parameters
alpha = 0.1
num_epochs = 10

W1, W2 = nnet(x, y, x_t, y_t, alpha, num_epochs)

f5 = open("p1q5.txt", "w+")
f6 = open("p1q6.txt", "w+")
f7 = open("p1q7.txt", "w+")
f8 = open("p1q8.txt", "w+")
f9 = open("p1q9.txt", "w+")

# output: 785 * 392
# print("Number of lines = " + str(len(W1)) + " Number of elements in a line = " + str(len(W1[0])))

for i in range(len(W1)):
    f5.write(str(round(W1[i][0], 4)))
    for j in range(1, len(W1[0])):
        f5.write(",")
        f5.write(str(round(W1[i][j], 4)))
    f5.write("\n")
f5.close()

# output: 1 * 392
# print("Number of lines = " + str(len(W2)) + " Number of elements in a line = " + str(len(W2[0])))
f6.write(str(round(W2[0][0], 4)))
for i in range(1, len(W2[0])):
    f6.write(",")
    f6.write(str(round(W2[0][i],4)))
f6.close()

# Todo: new test

new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0

B1 = W1[len(W1)-1]
W1 = np.delete(W1, len(W1)-1, 0)
B2 = W2[0][len(W2[0])-1]
W2 = np.delete(W2, len(W2[0])-1, 1)

# f = open("output.txt", "w+")
# np.savetxt("output.txt", B1, fmt="%2.3f")
# f.close()
# W1 dimensions 784 * 392
# B1 dimensions 392
# W2 dimensions 1 * 392
# B2 value 0.8409134652142897
# print("W1 dimensions " + str(len(W1)) + " * " + str(len(W1[0])))
# print("B1 dimensions " + str(len(B1)))
# print("W2 dimensions " + str(len(W2)) + " * " + str(len(W2[0])))
# print("B2 value " + str(B2))

a_1 = logis(np.matmul(new_x, W1) + B1)

z_2 = np.matmul(a_1, np.transpose(W2)) + B2

# f_test = open("output.txt", "w+")

# f_test.write(str(a_1[0][0]))
# for j in range(1, len(z_2)):
#     f_test.write(",")
#     f_test.write(str(z_2[j][0]))


a_2 = logis(z_2)

a_2_predict = np.round(a_2, decimals=1)

print(str(len(a_2)) + " * " + str(len(a_2[0])))
f7.write(str(round(a_2[0][0], 2)))
f8.write(str(round(a_2[0][0])))
for i in range(1, len(a_2)):
    f7.write(",")
    f7.write(str(round(a_2[i][0], 2)))
    f8.write(",")
    f8.write(str(round(a_2[i][0])))
f7.close()
f8.close()

flag = 0
index = -1
for i in range(200):
    if i < 100 and a_2_predict[i] != 0:
        flag = 1
        index = i
        break
    if i > 99 and a_2_predict[i] != 1:
        flag = 1
        index = i
        break
print(flag)
if (flag):
    f9.write(str(new_test[index][0]))
    for i in range(1, len(new_test[index])):
        f9.write(",")
        f9.write(str(new_test[index][i]))
f9.close()
    
