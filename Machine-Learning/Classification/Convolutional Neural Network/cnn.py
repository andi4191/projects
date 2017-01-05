#!/usr/bin/python

#Credits : www.tensorflow.org. Most of the APIs have been referenced from the tensorflow tutorials and Project Supplement

import tensorflow as tf
import numpy as np
#from sklearn.datasets import fetch_mldata
from tensorflow.examples.tutorials.mnist import input_data
import fnmatch
import os
import cv2
from PIL import Image


class CNN:
	def __init__(self, data, K, N):
		self.data = data
		self.K = K
		self.N = N
		
	
	def oneHotEncoding(self,t):
		target = np.zeros((self.K,1))
		target[t] = 1
		return target
	
	
	def build_model(self, x):
		
		#Taking 5*5 window for 1 pixel input and giving 32 output pixels 
		temp = tf.random_normal([5, 5, 1, 32])
		w_conv1 = tf.Variable(temp)
		
		#Taking 5*5 window for 32 inputs from conv_1 and giving 64 values
		temp1 = tf.random_normal([5, 5, 32, 64])
		w_conv2 = tf.Variable(temp1)
		
		#Feeding the output to the fully connected /densly connected layer
		tmp_fc = tf.random_normal([7*7*64, 1024])
		w_fclayer = tf.Variable(tmp_fc)
		
		tmp_y = tf.random_normal([1024, self.K])
		w_y = tf.Variable(tmp_y)
		
		
		b_c1 = tf.Variable(tf.random_normal([32]))  # Dimension same as output in the weights
		b_c2 = tf.Variable(tf.random_normal([64]))
		b_fc = tf.Variable(tf.random_normal([1024]))
		b_out = tf.Variable(tf.random_normal([self.K]))
		
		x = tf.reshape(x, shape=[-1, 28, 28, 1])  #Image is flattened in the dataset so making it 2D
		conv1 = tf.nn.relu(self.convolve(x, w_conv1) + b_c1)
		conv1 = self.pool(conv1)
		
		conv2 = tf.nn.relu(self.convolve(conv1, w_conv2) + b_c2)
		conv2 = self.pool(conv2)
		
		full_clayer = tf.reshape(conv2, [-1, 7*7*64])
		full_clayer1 = tf.nn.relu(tf.matmul(full_clayer, w_fclayer) + b_fc)
		
		y_predict = tf.matmul(full_clayer1, w_y) + b_out
		
		return y_predict 
	
	def pool(self, val):
		ret_val = tf.nn.max_pool(val, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		#ksize = kernel size for convolution. Used 2*2 in this case
		
		return ret_val
		
	def convolve(self,x, w):
		#strides = mving the window of size same as ksize across the image
		#padding = To convolve the image pixels at the edge. Enlarging the image array with padding same as boundary pixel values
		
		ret_val = tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME')
		return ret_val
		
		
	def start_seq(self, x, y_, X, t):
		
		out = self.build_model(x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))
		opt = tf.train.AdamOptimizer().minimize(cost)
		
		
		y_conv = self.build_model(x)
		
		epochs = 10
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			
			#print type(X),type(self.data.validation.images),type(t),type(self.data.validation.labels)
			
			#print (sess.run(X))
			#print (sess.run(tf.shape(X)))
			#print (sess.run(t))
			#print (sess.run(tf.shape(self.data.validation.images)))
			#print (sess.run(tf.shape(self.data.test.images)))
			for i in range(0,epochs):
				err = 0
				for k in range(int(self.data.train.num_examples/self.N)):
					new_x, new_y = self.data.train.next_batch(self.N)
					tmp, err_val = sess.run([opt, cost], feed_dict={x:new_x, y_:new_y})
					err += err_val
					
				print "epoch",i,"error",err/self.N
			match = tf.equal(tf.argmax(out, 1), tf.argmax(y_,1))
			acc = tf.reduce_mean(tf.cast(match, 'float'))
			
			#tr = acc.eval({x:self.data.train.images, y_:self.data.train.labels})
			#print "Error :",(1-tr),"Accuracy <Train Set>",100*tr,"%"
			
			va = acc.eval({x:self.data.validation.images, y:self.data.validation.labels})
			print "Error :",(1-va),"Accuracy <Validation Set>",100*va,"%"
			
			te = acc.eval({x:self.data.test.images, y_:self.data.test.labels})
			print "Error :",(1-te),"Accuracy <Test Set>",100*te,"%"
			
			print "USPS DataSet"
			inp = X[0:9999][:]
			lab = t[0:9999][:]
			usp = acc.eval(feed_dict={x:inp, y_:lab})
			print "Error :",(1-usp),"Accuracy :",100*usp,"%"
			
			
	def re_scale_img(self):
		
		files=[]
		for root, dirnames, filenames in os.walk('usps'):
			for filename in fnmatch.filter(filenames, '*.png'):
				
				
				#label = root[-1]
				label = filter(str.isdigit, root)
				if(label != ''):
					fnam = os.path.join(root, filename)
					files.append([fnam,int(label)])
                

		
		n_w = 28
		n_h = 28
		sz = len(files)
		res = np.zeros((sz,785))
		for idx,i in enumerate(files):
			
			img = Image.open(i[0])
			img = img.resize((n_w, n_h), Image.ANTIALIAS)
			a = np.asarray(img).ravel()
			res[idx,0:784] = a / np.sum(a)
			res[idx,784] = i[1]

		return res
	
	def split_data(self, data):
		
		X = data[:,0:784]
		t = data[:,784]
		
		return X,t
	
	def get_data(self):
		
		X, t = self.split_data(self.data)
		return X,t
		
	def evaluate_model(self,x, y, mnist):
		
		X, t = self.split_data(self.data)
		out = self.build_model(x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
		opt = tf.train.AdamOptimizer().minimize(cost)
		
		
		epochs = 1
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for i in range(0,epochs):
				err = 0
				print sess.run(mnist.test.images.eval()),"test shape"
				for k in range(int(mnist.train.num_examples/self.N)):
					new_x, new_y = mnist.train.next_batch(self.N)
					tmp, err_val = sess.run([opt, cost], feed_dict={x:new_x, y:new_y})
					err += err_val
					
				print "epoch",i,"error",err
			match = tf.equal(tf.argmax(out, 1), tf.argmax(y,1))
			acc = tf.reduce_mean(tf.cast(match, 'float'))
			#print "Accuracy",100*acc.eval({x:self.data.test.images, y:self.data.test.labels}),"%"
			print "Accuracy",100*acc.eval({x:X, y:t}),"%"
		
		
		
if __name__ == "__main__":
	
	print "########  Convolutional Neural Network Model  ########"
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	K = 10
	N = 100
	
	print "MNIST DataSet"
	x = tf.placeholder('float', [None, 784])
	y = tf.placeholder('float')
	cnn = CNN(mnist,K,N)
	
	dat = cnn.re_scale_img()
	np.random.shuffle(dat)
	usps = CNN(dat, K, N)
	X,tmp = usps.get_data()
	#print tmp
	l=len(X)
	tmp = np.transpose(np.asmatrix(tmp))
	t = np.zeros((l,10))
	for i in range(l):
		val = tmp[i][0]
		t[i][int(val)] = 1 
	
	
	inp = X
	label = t
	cnn.start_seq(x, y, inp, label)
	
	
	
	
