import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
raw = []

with open('FGL.csv', 'rb') as csvfile:
    rawIn = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rawIn:
        rawRow = row[0].split(',')
        curRow = []
        for value in rawRow:   
            if(value == ''):
                curRow.append(0)
            else:
                curRow.append(float(value))
        raw.append(curRow)      
data = np.array(raw)
np.random.shuffle(data)
X = data[:,0:-1]
Y = data[:,-1:]

X_train = X[0:4500]
Y_train = Y[0:4500]
X_test = X[4500:]
Y_test = Y[4500:]

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X = tf.placeholder(tf.float32, shape=(None, 19))
Y = tf.placeholder(tf.float32, shape=(None, 1))
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", [19, 100], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [100], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [100,100], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [100], initializer=tf.zeros_initializer())
W3 = tf.get_variable("W3", [100,50], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", [50], initializer=tf.zeros_initializer())
W4 = tf.get_variable("W4", [50,1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [1], initializer = tf.zeros_initializer())

X = tf.nn.l2_normalize(X)
Z1 = tf.matmul(X, W1)+b1 
A1 = tf.nn.relu(Z1)
Z2 = tf.matmul(A1, W2)+b2
A2 = tf.nn.relu(Z2)
Z3 = tf.matmul(A2, W3)+b3
A3 = tf.nn.relu(Z3)
Z4 = tf.matmul(A3, W4)+b4
A4 = tf.nn.relu(Z3)
Z4 = tf.matmul(A4, W4)+b4

def compute_cost(Z4, Y):
    cost = tf.reduce_mean((Z4-Y)**2)
    return cost

cost = compute_cost(Z4, Y)
starter_learning_rate = 0.0001
global_step = tf.Variable(0, trainable=True)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.85, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init=tf.global_variables_initializer()
outData = Z4

batch_size = 100
with tf.Session() as sess:
    sess.run(init)
    train_costs = []
    test_costs = []
    for epoch in range(450):
        for i in range(0, 4500, batch_size):
            sess.run(optimizer, feed_dict={X:X_train[i:i+batch_size], Y:Y_train[i: i+batch_size], keep_prob : 0.65})
        train_costs.append(sess.run(cost, feed_dict={X:X_train, Y:Y_train, keep_prob : 1}))
        test_costs.append(sess.run(cost, feed_dict={X:X_test, Y:Y_test, keep_prob : 1}))
        if epoch%10 == 9:
            print("Test costs after " + str(epoch+1)+ " epochs: " + str(train_costs[-1]))
    iterations = list(range(450))
    plt.plot(iterations, train_costs, label="Train")
    plt.plot(iterations, test_costs, label ='Test')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show

    accuracy = tf.reduce_mean((Z4-Y)**2)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob : 1})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob : 1})
    print("RMSE Train:", np.sqrt(train_accuracy))
    print("RMSE Test:", np.sqrt(test_accuracy))
        
    with open('current.csv', 'rb') as curCSV:
        curIn = csv.reader(curCSV, delimiter=',', quotechar='|')
        rawData = []    
        output = []
        names = []
        curSiera = []
        curERA = []  
        for row in curIn:
            if(row[0] == 'Name'):
                continue      
            curRow = []
            names.append(row[0])
            curSiera.append(row[-2])
            curERA.append(row[-1])        
            for value in row[1:]:
                if(value == ''):
                    curRow.append(0)
                else:
                    curRow.append(float(value))     
            rawData.append(curRow)
    data = np.array(rawData)
    finalOutData = sess.run(outData, feed_dict={X:data[:,:-2]})
    output = np.column_stack((np.asarray(names),np.asarray(curERA),np.asarray(curSiera),finalOutData))
    df = pandas.DataFrame(output)    
    df.to_csv("estimted.csv")
