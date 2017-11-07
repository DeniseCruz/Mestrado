import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from more_itertools import chunked

import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams.update({'figure.max_open_warning': 0})
print("Bibliotecas carregadas!")

path = 'cifar10\\data_batch_1'
if os.path.isfile(path):
    print("achou")
else:
    print("file not exists")
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#treino de 100 a 200
treino = unpickle(os.getcwd() + '\\cifar10\\data_batch_1')
teste = unpickle(os.getcwd() + '\\cifar10\\test_batch')

x_train = treino[b'data'][0:10000]
x_test = teste[b'data'][0:5000]
y_train = treino[b'labels'][0:10000]
y_test = teste[b'labels'][0:5000]

print(treino[b'data'].shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Convert class vectors to binary class matrices.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
    
#raw_float = np.array(x_train, dtype=float)/255
#raw_float = np.array(x_train, dtype=float)
# Reshape the array to 4-dimensions.
images = raw_float.reshape([-1, 3, 32, 32])
# Reorder the indices of the array.
x_train_images = images.transpose([0, 2, 3, 1])


tf.set_random_seed(777)
learning_rate = 0.001
quantidade_maxima_epocas= 10
batch_size = 100

#keep_prob = 0.5
#keep_prob = tf.placeholder(tf.float32)
 
#X = tf.placeholder(tf.float32, [None, 3072])
#X_img = tf.reshape(X, [-1, 32, 32, 3])
#Y_real = tf.placeholder(tf.float32, [None, 10])


X = tf.placeholder(tf.float32, [None, 32,32,3], name='Entrada_X')
Y_real = tf.placeholder(tf.float32, [None, 10], name='Saida_Y_Real')

 
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
 
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
 
#W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],initializer=tf.contrib.layers.xavier_initializer())

W4 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 512], stddev=0.1))

b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
#L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
 
#W5 = tf.get_variable("W5", shape=[625, 10],initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.Variable(tf.truncated_normal([512,10], stddev=0.1))
#W5 = tf.get_variable(W4, shape=[512, 10])
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5


custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_real))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)
 
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_real, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
#def predict(self, x_test, keep_prop=1.0):
#    return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})
 
#def get_accuracy(self, x_test, y_test, keep_prop=1.0):
#    return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})
 
#def train(self, x_data, y_data, keep_prop=0.7):
#    return self.sess.run([self.cost, self.optimizer], feed_dict={
#            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

print("Iniciando ....")

model = Sequential()
sess = tf.Session()
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
#models = []
#num_models = 4
#for m in range(num_models):
#    models.append(Model(sess, "model" + str(m)))
#sess.run(tf.global_variables_initializer())

# treina a rede
epoca=1
j=0
r_inicio=0
r_final=0
#keep_prob = tf.placeholder(tf.float32)

train_images = x_train_images
train_labels = y_train  

print('Rede inicialiada. Treinamento inicializado. Tome um cafe...')
    
lista_custo_epoca = []

train_images = train_images.reshape([-1, 32,32,3])
train_labels = train_labels.reshape([-1,10])

lista_train_image = list(chunked(train_images, 128))
lista_train_label = list(chunked(train_labels, 128))

print(len(lista_train_image))
print(len(lista_train_label))

for epoca in range(quantidade_maxima_epocas):
    custo_medio = []
               
    lista_train_image = list(chunked(train_images, 128))
    lista_train_label = list(chunked(train_labels, 128))
    
    for j in range(len(lista_train_image)):
        chunk_images = lista_train_image[j]
        chunk_labels = lista_train_label[j]  
                
        #plt.figure(figsize=(2,2))
        #plt.imshow(single_img_reshaped, interpolation='nearest')
        #plt.imshow((train_image.reshape(32,32,3)))
        # plt.show()

        #custo_resultado, _ = sess.run([custo, optimizer], feed_dict={x: train_images, y_real: train_labels, keep_prob: 0.5})
        custo_resultado, _ = sess.run([custo, optimizer], feed_dict={X: chunk_images, Y_real: chunk_labels})
     
        custo_medio.append(custo_resultado)

    print('Epoca:', '%04d' % (epoca + 1), 'perda =', '{:.9f}'.format(np.average(custo_medio)))
    custo_epoca = np.average(custo_medio)
    lista_custo_epoca.append(custo_epoca)
    
    epoca+= 1
    
print('Treinamento finalizado!')
    
    
#visualizar erro
plt.title("Perda")
plt.xlabel("epoca")
plt.ylabel("Custo")
plt.plot(lista_custo_epoca, c='darkblue')
plt.show()


print("Teste iniciando")
i=0

#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_real, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#acerto = sess.run(accuracy, feed_dict={X: test_image, Y_real: test_label, keep_prob: 1.0})

print("Calcula acurácia treino...")

chunked_train_images = list(chunked(train_images, 128))
chunked_train_labels = list(chunked(train_labels, 128))

accuracy_list = []

for i in range(len(chunked_train_images)):
    chunk_images = chunked_train_images[i]
    chunk_labels = chunked_train_labels[i]

    calculated_accuracy = accuracy.eval(session=sess, feed_dict={X: chunk_images, Y_real: chunk_labels
    #, keep_prob: KEEP_PROB_TEST
    })
    accuracy_list.append(calculated_accuracy)
    #print(calculated_accuracy)
    
acuracia_final = np.average(accuracy_list)
print("Acurácia do treino: " + str(acuracia_final))

print("Calcula acurácia teste...")

x_test = x_test.reshape([-1, 32,32,3])
y_test = y_test.reshape([-1,10])

chunked_test_images = list(chunked(x_test, 128))
chunked_test_labels = list(chunked(y_test, 128))

accuracy_list = []

for i in range(len(chunked_test_images)):
    chunk_images = chunked_test_images[i]
    chunk_labels = chunked_test_labels[i]

    calculated_accuracy = accuracy.eval(session=sess, feed_dict={X: chunk_images, Y_real: chunk_labels
    #, keep_prob: KEEP_PROB_TEST
    })
    accuracy_list.append(calculated_accuracy)
    #print(calculated_accuracy)
    
acuracia_final = np.average(accuracy_list)
print("Acurácia do teste: " + str(acuracia_final))

