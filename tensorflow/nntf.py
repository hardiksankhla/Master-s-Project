import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#set seed
#tf.set_random_seed(1234)

#reading the data set
def read_dataset():
    df = pd.read_csv("C:\\Users\\Hardik\\Desktop\\Masters_project\\tensorflow\\smd_hourly_2016 copy1.csv")
    print(len(df.columns))
    
    x1 = df[df.columns[0:1]].values
    x2 = df[df.columns[1:2]].values
    x3 = df[df.columns[2:3]].values
    x4 = df[df.columns[3:4]].values
    x5 = df[df.columns[4:5]].values
    x6 = df[df.columns[10:11]].values
    x7 = df[df.columns[11:12]].values
    
    y  = df[df.columns[5:6]].values
    print(x1.shape)
    
    print(y.shape)
    
    return(x1,x2,x3,x4,x5,x6,x7,y)
#def nomalize(df):
 #   for i in df.size[1]
  #      for j in df.size[0]
   #         df.[i][j] = df.[i][j]

#Read the data set    
x1,x2,x3,x4,x5,x6,x7,y = read_dataset()    

#Shuffle the dataset
x1,x2,x3,x4,x5,x6,x7,y = shuffle(x1,x2,x3,x4,x5,x6,x7,y,random_state=1)

#split the dataset into test and train parts
train_x1, test_x1,train_x2, test_x2,train_x3, test_x3,train_x4, test_x4,train_x5, test_x5,train_x6, test_x6,train_x7, test_x7, train_y, test_y = train_test_split(x1,x2,x3,x4,x5,x6,x7,y, test_size=0.2, random_state=415  )

print(train_x1.shape)
print(test_x1.shape)
print(train_y.shape)

#defining parameters
learning_rate =0.0013
training_epochs = 50000
n_dim =x1.shape[1]
print("n_dim",n_dim)
model_path ="C:\\Users\\Hardik\\Desktop\\Masters_project\\tensorflow\\"

#define no of hidden layers ans no of neurons in each layers
n_hidden_11=12
n_hidden_12=4
n_hidden_13=12
n_hidden_14=25
n_hidden_15=25
n_hidden_16=50
n_hidden_17=50

n_hidden_21=16
n_hidden_22=25
n_hidden_23=50

n_hidden_3=13
n_hidden_4=10

n_hidden_5=5



X1 = tf.placeholder(tf.float32,[None,n_dim])
X2 = tf.placeholder(tf.float32,[None,n_dim])
X3 = tf.placeholder(tf.float32,[None,n_dim])
X4 = tf.placeholder(tf.float32,[None,n_dim])
X5 = tf.placeholder(tf.float32,[None,n_dim])
X6 = tf.placeholder(tf.float32,[None,n_dim])
X7 = tf.placeholder(tf.float32,[None,n_dim])
Y_ = tf.placeholder(tf.float32,[None, 1])

W = tf.Variable(tf.zeros([n_dim,1]))
b = tf.Variable(tf.zeros([1]))

#define the model
def multilayer_nn(X1,X2,X3,X4,X5,X6,X7, weights, biases):
    layer_11 = tf.add(tf.matmul(X1, weights['h11']), biases['b11'])
    layer_11 = tf.nn.tanh(layer_11)

    layer_12 = tf.add(tf.matmul(X2, weights['h12']), biases['b12'])
    layer_12 = tf.nn.tanh(layer_12)

    layer_13 = tf.add(tf.matmul(X3, weights['h13']), biases['b13'])
    layer_13 = tf.nn.tanh(layer_13)

    layer_14 = tf.add(tf.matmul(X4, weights['h14']), biases['b14'])
    layer_14 = tf.nn.tanh(layer_14)

    layer_15 = tf.add(tf.matmul(X5, weights['h15']), biases['b15'])
    layer_15 = tf.nn.tanh(layer_15)

    layer_16 = tf.add(tf.matmul(X6, weights['h16']), biases['b16'])
    layer_16 = tf.nn.tanh(layer_16)

    layer_17 = tf.add(tf.matmul(X7, weights['h17']), biases['b17'])
    layer_17 = tf.nn.tanh(layer_17)


    
    layer_21 = tf.add( tf.add(tf.matmul(layer_12, weights['h2112']), biases['b2112']), tf.add(tf.matmul(layer_13, weights['h2113']), biases['b2113']))
    layer_21 = tf.nn.sigmoid(layer_21)

    layer_22 = tf.add( tf.add(tf.matmul(layer_14, weights['h2214']), biases['b2214']),tf.add(tf.matmul(layer_15, weights['h2215']), biases['b2215']))
    layer_22= tf.nn.sigmoid(layer_22)

    layer_23 = tf.add( tf.add(tf.matmul(layer_16, weights['h2316']), biases['b2316']),tf.add(tf.matmul(layer_17, weights['h2317']), biases['b2317']))
    layer_23= tf.nn.sigmoid(layer_23)



    layer_3 = tf.add( tf.add(tf.matmul(layer_22, weights['h322']), biases['b322']),tf.add(tf.matmul(layer_23, weights['h323']), biases['b323']) )
    layer_3 = tf.nn.tanh(layer_3)

    layer_4 = tf.add( tf.add(tf.matmul(layer_21, weights['h421']), biases['b421']),tf.add(tf.matmul(layer_3, weights['h43']), biases['b43']) )
    layer_4 = tf.nn.relu(layer_4)



    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    
    
    
    
    out_layer = tf.add(tf.matmul(layer_5, weights['out']), biases['out'])
    print('outputlayer shape')
    print(out_layer.shape)
    return out_layer
    
#defining the weights and biases of each layer

weights = {
    'h11': tf.Variable(tf.truncated_normal([n_dim,n_hidden_11])),
    'h12': tf.Variable(tf.truncated_normal([n_dim,n_hidden_12])),
    'h13': tf.Variable(tf.truncated_normal([n_dim,n_hidden_13])),
    'h14': tf.Variable(tf.truncated_normal([n_dim,n_hidden_14])),
    'h15': tf.Variable(tf.truncated_normal([n_dim,n_hidden_15])),
    'h16': tf.Variable(tf.truncated_normal([n_dim,n_hidden_16])),
    'h17': tf.Variable(tf.truncated_normal([n_dim,n_hidden_17])),
    
    'h2111': tf.Variable(tf.truncated_normal([n_hidden_11,n_hidden_21])),
    'h2112': tf.Variable(tf.truncated_normal([n_hidden_12,n_hidden_21])),
    'h2113': tf.Variable(tf.truncated_normal([n_hidden_13,n_hidden_21])),
    'h2214': tf.Variable(tf.truncated_normal([n_hidden_14,n_hidden_22])),
    'h2215': tf.Variable(tf.truncated_normal([n_hidden_15,n_hidden_22])),
    'h2316': tf.Variable(tf.truncated_normal([n_hidden_16,n_hidden_23])),
    'h2317': tf.Variable(tf.truncated_normal([n_hidden_17,n_hidden_23])),
    
    'h322': tf.Variable(tf.truncated_normal([n_hidden_22,n_hidden_3])),
    'h323': tf.Variable(tf.truncated_normal([n_hidden_23,n_hidden_3])),
    
    'h421': tf.Variable(tf.truncated_normal([n_hidden_21,n_hidden_4])),
    'h43': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    
    'h5': tf.Variable(tf.truncated_normal([n_hidden_4,n_hidden_5])),
    
    'out': tf.Variable(tf.truncated_normal([n_hidden_5,1]))
}
biases = {
    'b11': tf.Variable(tf.truncated_normal([n_hidden_11])),
    'b12': tf.Variable(tf.truncated_normal([n_hidden_12])),
    'b13': tf.Variable(tf.truncated_normal([n_hidden_13])),
    'b14': tf.Variable(tf.truncated_normal([n_hidden_14])),
    'b15': tf.Variable(tf.truncated_normal([n_hidden_15])),
    'b16': tf.Variable(tf.truncated_normal([n_hidden_16])),
    'b17': tf.Variable(tf.truncated_normal([n_hidden_17])),
    
    'b2111': tf.Variable(tf.truncated_normal([n_hidden_21])),
    'b2112': tf.Variable(tf.truncated_normal([n_hidden_21])),
    'b2113': tf.Variable(tf.truncated_normal([n_hidden_21])),
    'b2214': tf.Variable(tf.truncated_normal([n_hidden_22])),
    'b2215': tf.Variable(tf.truncated_normal([n_hidden_22])),
    'b2316': tf.Variable(tf.truncated_normal([n_hidden_23])),
    'b2317': tf.Variable(tf.truncated_normal([n_hidden_23])),

    'b322': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b323': tf.Variable(tf.truncated_normal([n_hidden_3])),
    
    'b43': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'b421': tf.Variable(tf.truncated_normal([n_hidden_4])),
    
    'b5': tf.Variable(tf.truncated_normal([n_hidden_5])),
    
    'out': tf.Variable(tf.truncated_normal([1])),
}


# saver object to save the model
saver = tf.train.Saver()

#Call the model
Y = multilayer_nn(X1,X2,X3,X4,X5,X6,X7, weights, biases)
print(Y.shape)
print(Y_.shape)
#Loss
squared_delta = tf.square(Y-Y_)
print(squared_delta.shape)
loss = tf.reduce_sum(squared_delta)
loss = loss/train_x1.shape[0]
#Optimize
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
#Accuracy by segmentaion
accuracy1 = tf.equal(tf.round(Y/2000), tf.round(Y_/2000))
accuracy = tf.reduce_sum(tf.cast(accuracy1, tf.float32 ))
accuracy = accuracy/train_x1.shape[0]

#Accuracy by percentage
acc_by_per = tf.abs((Y-Y_)/Y)
accuracybyper = tf.reduce_sum(acc_by_per)
accuracybyper = (accuracybyper/train_x1.shape[0])*100

# initialize the variable
init = tf.global_variables_initializer()

#session run

sess = tf.Session()
#File_Writer = tf.summary.FileWriter('C:\\Users\\Hardik\\Desktop\\Masters_project\\tensorflow\\graph', sess.graph)
sess.run(init)

File_Writer = tf.summary.FileWriter('C:\\Users\\Hardik\\Desktop\\Masters_project\\tensorflow\\graph', sess.graph) 
loss_history =[]
final_accuracy =[]
final_acc =[]
final_a =[]
for i in range (training_epochs):
    sess.run(train,{X1:train_x1,X2:train_x2,X3:train_x3,X4:train_x4,X5:train_x5,X6:train_x6,X7:train_x7,Y_:train_y} )
    losses = sess.run(loss,{X1:train_x1,X2:train_x2,X3:train_x3,X4:train_x4,X5:train_x5,X6:train_x6,X7:train_x7,Y_:train_y})
    accu = sess.run(accuracy,{X1:train_x1,X2:train_x2,X3:train_x3,X4:train_x4,X5:train_x5,X6:train_x6,X7:train_x7,Y_:train_y})
    acc = sess.run(accuracybyper,{X1:train_x1,X2:train_x2,X3:train_x3,X4:train_x4,X5:train_x5,X6:train_x6,X7:train_x7,Y_:train_y})
    print(losses)
    print (accu)
    print (acc)
    loss_history = np.append(loss_history,losses)
    final_accuracy = np.append(final_accuracy,accu)
    final_acc = np.append(final_acc,acc)
    final_a = np.append(final_a,max(0,100-acc))
        #print(sess.run([w,b]))
      
save_path = saver.save(sess, model_path)

plt.plot(loss_history)
plt.show()
plt.plot(final_accuracy)
plt.show()
plt.plot(final_acc)
plt.show()
plt.plot(final_a)
plt.show()
print("final")
print(sess.run(loss,{X1:train_x1,X2:train_x2,X3:train_x3,X4:train_x4,X5:train_x5,X6:train_x6,X7:train_x7,Y_:train_y}))
#print(sess.run([w,b]))
