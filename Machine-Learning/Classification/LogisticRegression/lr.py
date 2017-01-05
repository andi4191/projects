from __future__ import division
import numpy as np
import cPickle
import gzip
import fnmatch
import os
import cv2
from PIL import Image


class LogisticReg():
	
	def __init__(self, train_d, val_d, test_d, dim, cls, w ,b):
		
		self.train = train_d
		self.val_d = val_d
		self.test_d = test_d
		self.ndim = dim
		self.classes = cls
		self.w = w
		self.b = b
			
	def split_data(self, data):
		
		X = data[:,0:self.ndim]
		t = data[:,self.ndim]
		return X,t
	
	def oneHotEncoding(self,t):
		
		target = np.zeros((self.classes,1))
		target[int(t)] = 1
		return target
	
	
	
	
	def iter_wt(self, x, t):
		
		eta = 0.0001
		reg_lambda = 0.01
		ak = np.dot(x,self.w)
		ak = np.add(ak,self.b)
		
		y = np.exp(ak)
		
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
	
		diff = y - T
		grad_E = np.dot(np.transpose(diff),x)
		
		#db1 = delta_j
		#self.b1 = np.subtract(self.b1,reg_lambda*eta*db1)
		self.w = np.subtract(self.w,eta*np.transpose(grad_E))
		
		return err
		
	def get_error(self,tvec,y):
		err = 0
		for i in range(0,self.classes):
			err += tvec[i]*np.log(y[i])
		err = -err
		return err
		
	def train_model(self, N):
		
		epochs = 200
		for k in range(0,epochs):
			
			X,t = self.split_data(self.train)
			eta = 0.001
		
			error = 0
			t = np.transpose(np.asmatrix(t))	
			it = len(self.train)//N
		
			for i in range(0,it):
			
				x = X[(i*N):(i*N+N)][:]
				tar = t[(i*N):(i*N+N)][:]
				err = self.iter_wt(x,tar)
			
			err_val = self.test_model("VALIDATION", N)	
			acc_val=  (1-err_val)*100
			if(k%10 == 0):
				print "epoch",k,"out of",epochs,"Classification Error :",err_val,"Accuracy <Validation dataset> :",acc_val,"%"
		return self.w, self.b	
	
	
	def test_model(self, FLAG,N):
		
		data = self.test_d
		if(FLAG=="VALIDATION"):
			data = self.val_d
		elif(FLAG=="TRAIN"):
			data = self.train
		X,t = self.split_data(data)
		
		err = 0
		sz = len(data)
		it = sz//N

		t = np.transpose(np.asmatrix(t))
		for i in range(0,it):

			x = X[(i*N):(i*N+N)][:]
			tar = t[(i*N):(i*N+N)][:]
			err += self.evaluate_model(tar, x, N)
		
		return err/len(data)
	
	def evaluate_model(self,t, x, N):
		
		#print "x.shape",x.shape,"w_shape",self.w.shape
		ak = np.dot(x,self.w)
		ak = np.add(ak,self.b)
		
		y = np.exp(ak)
		
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
	
	print "######## Logistic Regression Model ########"
	
	filename = 'mnist.pkl.gz'
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	tr_d = np.column_stack((training_data[0],training_data[1]))
	val_d = np.column_stack((validation_data[0],validation_data[1]))
	test_d = np.column_stack((test_data[0],test_data[1]))
	np.random.shuffle(tr_d)
	np.random.shuffle(val_d)
	np.random.shuffle(test_d)
	
	N = 1000
	D = 784
	K = 10
	
	w = np.random.random((D,K))
	w = w/np.sum(w) 
	
	bias = np.ones((N,K))
	print tr_d.shape
	lr = LogisticReg(tr_d,val_d,test_d, D, K, w, bias)
	
	print "MNIST Dataset"	
	wt, b_usps = lr.train_model(N)
	#acc_train=lr.test_model(w,tr_d,bias)
	err_train=lr.test_model("TRAIN", N)
	acc_train= (1-err_train)*100
	print "Classification Error :",err_train,"Accuracy <Train Set> :",acc_train,"%"
	err_test=lr.test_model("TEST", N)
	acc_test = (1-err_test)*100	
	print "Classification Error :",err_test,"Accuracy <Test Set> :",acc_test,"%"
	f.close()
	
	wt_usps = wt
	N = 1000
	X = lr.re_scale_img()	
	np.random.shuffle(X)
	test_data = X
	usps = LogisticReg(0,0,test_data, D, K, wt_usps, b_usps)
	print "USPS Dataset"
	err_test = usps.test_model("TEST",N)
	acc_test = (1-err_test)*100	
	print "Error :",err_test,"Accuracy <Test Set> :",acc_test,"%"
	 
	
