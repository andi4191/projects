from __future__ import division
import numpy as np
import gzip
import cPickle
import fnmatch
import os
import cv2
from PIL import Image


class SNN():
	
	def __init__(self,train_d, val_d, test_d, D,K,M,w1,w2,b1,b2):
		
		self.train = train_d
		self.val = val_d
		self.test = test_d
		self.D = D
		self.K = K
		self.M = M
		self.w1 = w1
		self.w2 = w2
		self.b1 = b1
		self.b2 = b2
	
	def sigmoid(self, x):
		
		val = 1 / (1 + np.exp(-x))
		return val;
	
	
	
	def split_data(self, data):
		
		X = data[:,0:self.D]
		t = data[:,self.D]
		return X,t

	def oneHotEncoding(self,t):
		target = np.zeros((self.K,1))
		target[t] = 1
		return target


	
	def calc_zval(self, w, x, b, j, D):
		
		for i in range(0,D):
			val = np.dot(w[j,i],x[i,:])
		
		val += b[j]
		return val
		
		
	def calc_act(self, w, z, b, j):
		
		for i in range(0, M):
			val = np.dot(w[j,i],z[j])
		
		act = val + b[j]
		return act 
	
	def get_derivative_h(self, x):
		
		return self.sigmoid(x)*(1-self.sigmoid(x))
		
	def compute_wt(self, t, M, D, x, K,N):
		
		eta = 0.0001
		reg_lambda = 0.01
		#0.01		
		layer_1 = np.dot(x,self.w1)
		
		layer_1 = np.add(layer_1, self.b1)
		z = self.sigmoid(layer_1)
		
		act = np.dot(z,self.w2)
		act = np.add(act, self.b2)
		
		#y = self.sigmoid(act)
		y = np.exp(act)
		
		for i in range(0,N):
			y[i]=y[i]/y[i].sum()
		
		Y = np.zeros((N,K),dtype='int')
		T = np.zeros((N,K),dtype='int')
		
		for i in range(0,len(y)):
			val = t[i][0]
			Y[i][np.argmax(y[i][:])] = 1
			T[i][int(val)] = 1
		
		
		tmp = np.logical_and(Y,T)
		count = np.sum(tmp.any(True))
		
		err = N-count
		#print count,"match"
		
		delta_k = y - T
		der_val = self.get_derivative_h(z)
		
		w_val = np.dot(self.w2, np.transpose(delta_k))
		
		delta_j = np.multiply(der_val,np.transpose(w_val))
		db1 = delta_j
		db2 = delta_k
		#delta_j = np.multiply(w_val, np.transpose((1 - np.power(z,2))))
		
		self.b1 = np.subtract(self.b1,reg_lambda*eta*db1) 
		self.b2 = np.subtract(self.b2,reg_lambda*eta*db2)
		
		grad_EW1 = reg_lambda * np.dot(np.transpose(x),delta_j)
		grad_EW2 = reg_lambda * np.dot(np.transpose(z),delta_k)
		
		self.w1 = np.subtract(self.w1,eta*grad_EW1)
		self.w2 = np.subtract(self.w2,eta*grad_EW2)
		
		return err
		
	
	def evaluate_model(self, t, M, D, x, K,N):
		
		
		layer_1 = np.dot(x,self.w1) + self.b1
		z = self.sigmoid(layer_1)
		
		act = np.dot(z,self.w2) + self.b2
		
		y = np.exp(act)
		
		for i in range(0,N):
			y[i]=y[i]/y[i].sum()
		
		Y = np.zeros((N,K),dtype='int')
		T = np.zeros((N,K),dtype='int')
		
		for i in range(0,len(y)):
			val = t[i][0]
			Y[i][np.argmax(y[i][:])] = 1
			T[i][int(val)] = 1
		
		
		tmp = np.logical_and(Y,T)
		count = np.sum(tmp.any(True))
		
		err = N-count
		
		return err
		
		
		
	def train_model(self, w1, w2,N):
		epochs = 100
		vt = []
		for k in range(0,epochs):
			
			err = 0
			sz = len(self.train)
			it = sz//N
			
			np.random.shuffle(self.train)
			X,t = self.split_data(self.train)
			t = np.transpose(np.asmatrix(t))
		
			for i in range(0,it):
					
				x = X[(i*N):(i*N+N)][:]
				tar = t[(i*N):(i*N+N)][:]
				err = self.compute_wt(tar, M, D, x, K, N)
			
			w1 = self.w1	
			w2 = self.w2
			
			
			err_val = snn.test_model("VALIDATION", w1, w2, N)
			acc_val = (1-err_val)*100
			if(k%5 == 0):
				print "epoch",k,"Error",err_val,"Accuracy <Validation Set>", acc_val,"%"
			
		
		return w1,w2,err/N,self.b1, self.b2
		
		
	def test_model(self, FLAG, w1, w2, N):
		data = self.test
		if(FLAG == "VALIDATION"):
			data = self.val
		
		X,t = self.split_data(data)
		err = 0
		sz = len(data)
		it = sz//N
		
		t = np.transpose(np.asmatrix(t))
		for i in range(0,it):
			
			x = X[(i*N):(i*N+N)][:]
			tar = t[(i*N):(i*N+N)][:]
			err += self.evaluate_model(tar, M, D, x, K, N)
		
		
		return err/len(data)
		
		
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
	
if __name__=="__main__":
	
	print "##########  Single Layer Neural Network Model  ###########"
	
	filename = 'mnist.pkl.gz'
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	
	tr_d = np.column_stack((training_data[0],training_data[1]))
	val_d = np.column_stack((validation_data[0],validation_data[1]))
	test_d = np.column_stack((test_data[0],test_data[1]))
	np.random.shuffle(tr_d)
	np.random.shuffle(val_d)
	np.random.shuffle(test_d)
	
	D = 784
	K = 10
	N = 10000
	#Since single layer its ok to keep it static for first layer
	M = 1600
	#M = 200
	
	w1 = np.random.randn(D,M)
	w1 = w1/np.sum(w1)
	w2 = np.random.randn(M,K)
	w1 = w1/np.sum(w2)
	
	
	b1 = np.ones((M,1))
	b2 = np.ones((K,1))
	
	
	
	b1_ = np.zeros((N,M))
	b1_[range(N)] = np.transpose(b1)
		
	b2_ = np.zeros((N,K))
	b2_[range(N)] = np.transpose(b2)
	
	b1 = b1_
	b2 = b2_
	
	#snn = SNN(training_data, validation_data, test_data, D, K, M,w1,w2,b1,b2)
	snn = SNN(tr_d, val_d, test_d, D, K, M,w1,w2,b1,b2)
	w1,w2,err_train, rb1, rb2 = snn.train_model(w1, w2, N) 
	print "Classification Error",err_train,"Accuracy <Training Set>", (1-err_train)*100,"%"
	
	err_val = snn.test_model("VALIDATION", w1, w2, N)
	print "Classification Error",err_val,"Accuracy <Validation Set>", (1-err_val)*100,"%"
	
	err_test = snn.test_model("TEST", w1, w2, N)
	print "Classification Error",err_test,"Accuracy <Test Set>", (1-err_test)*100,"%"
	
	############  USPS Data Set  #####################
	
	
	wt_usps1 = w1
	wt_usps2 = w2
	
	X = snn.re_scale_img()	
	np.random.shuffle(X)
	test_data = X
	usps = SNN(0, 0, test_data, D, K, M, wt_usps1,wt_usps2,rb1,rb2)
	print "USPS Dataset"
	err_test = usps.test_model("TEST", wt_usps1, wt_usps2, N)
	acc_test = (1-err_test)*100	
	print "Accuracy <Test Set> :",acc_test,"%"
	
	
