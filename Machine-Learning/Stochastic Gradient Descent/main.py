##############################################################################
#
#      Date             Author              Description
#      22-Oct-2016     Anurag Dixit        Initial draft (UBITName: anuragdi)
#
##############################################################################

import numpy as np
import math
import random as rnd
#from scipy.cluster.vq import kmeans
#from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt
import csv

#Data class is used to parse and store the data extracted from the seed excel file

class Data(object):
	
	
	def __init__(self,dim):
		
		self.filename=0
		self.dim=dim
		self.num_lines = 0
		self.num_lines_train=0
		self.num_lines_test=0
		self.num_lines_val=0
		self.train = 0
		self.validate = 0
		self.test = 0
		self.target = 0
		self.mus=0
	
	def get_file_len(self,f):
		
		with open(self.filename) as f:
			for i, l in enumerate(f):
				pass
		return i + 1
	
	def get_train(self):
		
		return self.train
	
	def parse_real_data(self,fname,M):
		self.filename=fname
		dim=self.dim
		feature=[]
		target=[]
		tmp=[]
		f=open(self.filename,'r')
		num_lines=self.get_file_len(f)
		self.num_lines=num_lines
		for r in range(0,num_lines):
			ftrEle=[]
			dat=f.readline().split(" ")
			target.append(dat[0])
			for i in range(2,dim+2):
				d=dat[i].split(':')
				ftrEle.append(d[1])
			tmp = [x for x in ftrEle]
		
			feature.append(tmp)
		
		ft=np.zeros((num_lines,dim+1),dtype='float')
		t=np.zeros((num_lines,1),dtype='float')
		for idx,i in enumerate(feature):
			i.append(target[idx])
			#print i
			ft[idx]=i
			#t[idx]=target[idx]
		np.random.shuffle(ft)
		target=ft[:,dim]
		feature=ft[:,0:dim]
		
		
		
		train_count=int(math.floor(num_lines*0.8))
		self.num_lines_train=train_count
		val_count=int(math.floor(num_lines/10))
		self.num_lines_val=val_count
		test_count=num_lines-(train_count + val_count)
		self.num_lines_test=test_count
		
		self.train = np.array(feature[0:train_count],dtype='float')
		self.validate = np.array(feature[train_count:train_count+val_count],dtype='float')
		self.test = np.array(feature[train_count+val_count:],dtype='float')
		self.target=np.array(target[:],dtype='int')
		
		#print "andi"
		#ele=np.array(a[:,6],dtype='float')
		#print ele
		#print np.var(a[:,1])
		#print feature.shape
		'''
		kmeans = KMeans(n_clusters=M)
		kmeans.fit(self.train)

		centroid=kmeans.cluster_centers_
		
		mu=[]
		
		for i in range(0,len(centroid)):
			tmp=[]
			mus=[]
			for j in range(0,dim):
				row=centroid[i][j]
				mus.append(row)
			tmp=[x for x in mus]
			mu.append(tmp)
		
		self.mus=np.array(mu,dtype='float')
		'''
		'''
		for i in range(0,M):
			a1=self.train[:,i]==0
			a2=self.validate[:,i]==0
			a3=self.test[:,i]==0
			anyNonzero1=any(a1==False)
			anyNonzero2=any(a2==False)
			anyNonzero3=any(a3==False)
			if(True==anyNonzero1):
				self.train[:,i]=0.0001
			if(True==anyNonzero1):
				self.validate[:,i]=0.0001
			if(True==anyNonzero1):
				self.test[:,i]=0.0001
		'''
		f.close();
	
	def parse_syn_data(self,fname_in,fname_out,M):
		
		fin=open(fname_in,'rb')
		data = [row for row in csv.reader(fin.read().splitlines())]
		data_size=len(data)
		
		tr_size=int(np.floor(0.8*data_size))
		val_size=int(np.floor(0.1*data_size))
		test_size=int(np.floor(data_size-tr_size-val_size))
		self.num_lines=tr_size+val_size+test_size
		num_lines=self.num_lines
		#print self.num_lines
		self.num_lines_train=tr_size
		self.num_lines_val=val_size
		self.num_lines_test=test_size
		#dat=list()
			
		data1=np.genfromtxt(fname_out,delimiter=',')
		
		data_size1=len(data1)
		
		target=[]
		for i in range(0,data_size1):
			num=data1[i]
			#print num
			#num=np.array(num,dtype='float')
			target.append(num)
		
		ft=np.zeros((num_lines,dim+1),dtype='float')
		#t=np.zeros((num_lines,1),dtype='float')
		for idx,i in enumerate(data):
			
			i.append(target[idx])
			ft[idx]=i
			
		np.random.shuffle(ft)
		target=ft[:,dim]
		data=ft[:,0:dim]
		
		
		
			
		self.train=np.array(data[0:tr_size],dtype='float')
		self.validate=np.array(data[tr_size:tr_size+val_size],dtype='float')
		self.test=np.array(data[tr_size+val_size:],dtype='float')
		fin.close()
		
		self.target=np.array(target,dtype='float')
		
		#fout.close()
			
class LinearReg(Data):
	
	def __init__(self,dim):
		Data.__init__(self,dim)
	
	def get_target(self,get_data_set):
		
		complete_target=self.target[:len(self.train)+len(self.validate)+len(self.test)]
		ret_target=0
		
		if(get_data_set=="TEST"):
			st_offset=self.num_lines_train+self.num_lines_val
			ret_target=complete_target[st_offset:st_offset+self.num_lines_test]
		elif(get_data_set=="VALIDATION"):
			st_offset=self.num_lines_train
			ret_target=complete_target[st_offset:st_offset+self.num_lines_val]
		elif(get_data_set=="TRAIN"):
			st_offset=0
			ret_target=complete_target[st_offset:st_offset+self.num_lines_train]
		else:
			print "Incorrect data set selected for retrieving target values"
		
		
		ret_val=np.array(ret_target,dtype='int')
		return ret_val

	
	def get_phi(self,diff,dim,sigma_inv):
		
		tmp1=np.dot(diff.transpose(),sigma_inv)
		retVal=-0.5*np.dot(tmp1,diff)
		
		retVal=np.exp(retVal)
		
		return retVal
				
		
	def basis_func(self,get_data_set,M):
		
		mu_temp=[]
		dim=self.dim
		num_lines=0
		num_data_set=0
		data_set=0
		if(get_data_set=="TRAIN"):
			num_lines=self.num_lines_train
			num_data_set=len(self.train)
			data_set=self.train
		elif(get_data_set=="TEST"):
			num_lines=self.num_lines_test
			num_data_set=len(self.test)
			data_set=self.test
		elif(get_data_set=="VALIDATION"):
			num_lines=self.num_lines_val
			num_data_set=len(self.validate)
			data_set=self.validate
		
		rand_row=[]
		phis=[]
		basis_phi=[]
		
		feature_mus=[]
		mus=[]
		
		
		#num_x_train=len(self.train)
		for i in range(M):
			#Take random data points for mu(j) from each batch within training set
			k=rnd.randint(i*(num_data_set/M),(i+1)*(num_data_set/M)) 
			rand_row.append(k)
		
		for i in rand_row:
			tmp=[]
			for j in range(0,dim):
				row=data_set[i]
				mus.append(row[j])
			tmp = [x for x in mus]
			feature_mus.append(tmp)
		
		#Construct the mu matrix  (M by dim) dimension
		mu_mat=[]
		
		
		for i in range(0,M):
			
			mu_mat.append(data_set[rand_row[i],:])
		
		
		
		'''
		#Construct the mu matrix  (M by dim) dimension
		
		mu_mat=[]
		for i in range(0,len(self.mus)):
			mu_mat.append(self.mus[i])
		
		'''
		#Construct the sigma inv (dim*dim) matrix
		
		sigmas=[]
		sig=[]
		for i in range(0,dim):
			sig.append([])
		
		for i in range(0,dim):
			
			for j in range(0,dim):
				if(i!=j):
					sig[i].append(0)
				else:
					feature=zip(*data_set)[i]
					if(np.var(feature)==0):
						sig[i].append(0.001)
					else:
						sig[i].append(0.1*(np.var(feature)))
					
					
				tmp=[x for x in sig[i]]
				
			sigmas.append(tmp)
		
		
		sigma=np.array(sigmas,dtype='float')
		
		try:
			sigma_inv=np.linalg.inv(sigma)
			#sigma_inv=np.identity(dim)
		except np.linalg.linalg.LinAlgError as err:
			if 'Singular matrix' in err.message:
				sigma_inv=np.identity(dim)
				pass
		
		#Construct the phi_mat of (50k by M) dimensions
		
		#Making it a multi dimensional list
		s=[]
		t=[]
		
		for i in range(0,num_lines):
			phis.append([])
		for i in range(0,num_lines):
			l=[]
			s=[]
			x_vec=data_set[i]
			for j in range(0,M):
				if(j==0):
					val=1
				else:
					curr_mu=mu_mat[j]
					diff=[i-j for i,j in zip(x_vec,curr_mu)]
					mod_diff=np.array(diff,dtype='float')
					val=self.get_phi(mod_diff,dim,sigma_inv)
					
					#phis[i].append(val)
				l.append(val)
				s=[x for x in l]
			t.append(s)
			
		
		phi=np.array(t,dtype='float')	
		#print "Design matrix"
		#print phi
		
		return phi
	
	def closed_form_rbf(self,reg_lambda,phi,M):
		#M=self.M
		
		t=self.target[:len(self.train)]
		I=np.identity(M)
		tmp1=np.dot(phi.transpose(),phi)
				
		try:
			tmp2=np.linalg.inv((reg_lambda*I)+tmp1)
		except np.linalg.linalg.LinAlgError as err:
			if 'Singular matrix here' in err.message:
				print "Singular matrix here"
				tmp2=np.linalg.inv(tmp1)
				pass
		#tmp2=np.linalg.inv((reg_lambda*I)+tmp1)
		'''
		print "tmp2"
		print tmp2.shape
		print tmp2
		'''
		tmp3=np.dot(tmp2,phi.transpose())
		'''
		print "tmp3 shape"
		print tmp3.shape
		print tmp3
		print "t shape"
		print t.shape
		'''
		w=np.dot(tmp3,t)
		'''
		print "w shape"
		print w.shape
		print w
		'''
		return w
	
	def get_Erms(self,trained_wt,reg_lambda,DATASET,test_phi):
		
		#Need to calculate the phi for the Test Dataset to calculate the error in prediction
		
		target_test=self.get_target(DATASET)
		
		Ed=self.get_Ed(trained_wt,test_phi,target_test)
		Ew=self.get_Ew(trained_wt)
		sz=0
		if(DATASET=="TRAIN"):
			sz=self.num_lines_train
		elif(DATASET=="VALIDATION"):
			sz=self.num_lines_val
		elif(DATASET=="TEST"):
			sz=self.num_lines_test
		E=Ed+reg_lambda*Ew;
		#E=np.dot(phi.transpose(),trained_wt)-target_test
		'''
		print "Ed: "+str(Ed)
		print "Ew: "+str(Ew)
		print "E "+str(E)
		print "sz: "+str(sz)
		'''
		#print "len target"+str(len(target_test))
		Erms=np.sqrt((2*E)/sz);
		return Erms;
	
	def get_Ed(self,trained_wt,test_phi,target):
		
		sum_val=0
		for i,t in enumerate(target):
			t_phi=test_phi[i,:]
			err=np.power((t-np.dot(trained_wt.transpose(),t_phi)),2)
			#print err
			sum_val=sum_val+err
		
		'''
		val1=np.dot(test_phi,trained_wt)
		print val1.shape
		val2=np.subtract(val1,target)
		retVal=np.dot(val2.transpose(),val2)
		
		return retVal #0.5*sum_val
		'''
		return 0.5*sum_val
		
	def get_Ew(self,trained_wt):
		return 0.5*np.dot(trained_wt.transpose(),trained_wt)
	
	def tune_parameters(self):
		
		wt=[]
		m_basis=[]
		e_rms=[]
		r_lambda=[]
	
		
		min_erms=99999
		train_wt=[]
		optimal_reg=0
		
		m_idx=0
		r_idx=0
		#for i in range(1,10):
		for i in range(1,25,5):
			M=i
			#phi=self.basis_func("TRAIN",M)
			phi=self.basis_func("TRAIN",M)
			val_phi=self.basis_func("VALIDATION",M)  
			for j in range(1,10):
				
				reg_lambda=j*0.1
				trained_wt=self.closed_form_rbf(reg_lambda,phi,M)
				#print "Weight trained: " 
				#print trained_wt
				Erms=self.get_Erms(trained_wt,reg_lambda,"VALIDATION",val_phi);
				wt.append(trained_wt)
				m_basis.append(i)
				r_lambda.append(reg_lambda)
				e_rms.append(Erms)
				#print "Erms over prediction: "+str(Erms)
				if(min_erms>Erms):
					m_idx=i
					r_idx=j
					train_wt=trained_wt
					#print "Updating training weigth: "
					#print train_wt
					min_erms=Erms
					optimal_reg=reg_lambda
				
		fig=plt.figure()
		plt.plot(m_basis,e_rms,label="M basis v/s Erms")
		plt.legend()
		plt.show()
		fig=plt.figure()
		plt.plot(r_lambda,e_rms,label="R lambda v/s Erms")
		plt.legend()
		plt.show()
		
		print m_basis
		print e_rms
		print r_lambda
		
		print "Optimal M: "+str(m_idx)
		print "Optimal reg_lambda: "+str(optimal_reg)
		print "Weights: "
		print train_wt
		print "(Validation) Erms: "+str(Erms)
		return {'RL':optimal_reg,'M':m_idx}

	
	def train_rbf(self,reg_lambda,M):
		
		phi=self.basis_func("TRAIN",M)
		val_phi=self.basis_func("VALIDATION",M)  
		test_phi=self.basis_func("TEST",M)
		
		trained_wt=self.closed_form_rbf(reg_lambda,phi,M)
		#print "Weights: "
		#print trained_wt
		Erms_train=self.get_Erms(trained_wt,reg_lambda,"TRAIN",phi);
		#print "(Training) Erms: "+str(Erms)
		Erms_val=self.get_Erms(trained_wt,reg_lambda,"VALIDATION",val_phi);
		#print "(Validation) Erms: "+str(Erms)
		
		#print test_phi[0]
		Erms_test=self.get_Erms(trained_wt,reg_lambda,"TEST",test_phi)
		#print "(Test Set) Erms: "+str(test_erms)
		#return {'TEST_PHI':phi,'VAL_PHI':val_phi}
		return Erms_train,Erms_val,Erms_test, phi, val_phi,trained_wt
		
class SGD(Data):
	
	def __init__(self,dim):
		Data.__init__(self,dim)
		
	
	def get_ed_sgd(self,target_vec,wt_vec,phi,N):
		sum_val=0
		for i in range(0,N):
			error=target_vec[i]-np.dot(wt_vec.transpose(),phi[i])
			err=np.array(error,dtype='float')
			sum_val=sum_val+err*phi[i]
		
		sum_val=-sum_val
		
		return sum_val	#/N
		
	def get_error_sgd(self,wt,target,phi,reg_lambda):
		
		N=self.num_lines_train
		ew=wt
		ed=self.get_ed_sgd(target,wt,phi,N)
		
		return ed+reg_lambda*ew
	
	
			
	
	
	def l2_norm(self,v):
		s=0
		for i in range(len(v)):
			s=s+pow(v[i],2)
		return np.sqrt(s)
		
	def compute_sgd(self,reg_lambda,M,phi):
		
		#Arbitrarily choosing initial weights from a uniform distributionX = np.array([[1, 2], [1, 4], [1, 0],
		
		w_ini=np.random.random(M)
		train_count=self.num_lines_train
		minima_reached=False
		DATASET="TRAIN"
		eta=1	# Learning rate for SGD (arbitrarily)
		MAX_ATTEMPT=1
		count=0
		min_err=99999
		target=self.target[:len(self.train)]
		curr_wt=w_ini	#(46*1) dimension matrix
		prev_error=10
		grad_ed=0
		grad_ew=0
		prev_wt=w_ini
		min_e_rms=99999
		wt_res=[]
		retract_count=0
		curr_weight=0
		wt=0
		
		#while (True!=minima_reached or i>=10):#self.num_lines_train):
		for i in range(0,100):#self.num_lines_train):
			
			#Compute predicted values
			curr_phi=phi[i]
			pred_val=np.dot(curr_phi,curr_wt)
		
			loss=target[i]-pred_val
			gradient_ed=-loss*curr_phi    #(M*1) dimension vector
			curr_error=self.get_Erms(curr_wt,reg_lambda,DATASET,phi)
			#curr_error=self.get_Erms(curr_wt,reg_lambda,"OTHERS",phi)
			
			diff=prev_error-curr_error
			gradient_ew=curr_wt
			grad=np.add(gradient_ed,reg_lambda*gradient_ew)
			
			if(min_err>curr_error):
				min_err=curr_error
				wt=curr_wt
			print "SGD Adpative Learning Current Error: "+str(curr_error)+" eta: "+str(eta)
			#If error decreases then accept else reduce the eta by a factor of 2
			if(curr_error>=prev_error):
				eta=eta/2
				#grad=-grad
				continue
				
			
			
			
			delta_wt=-eta*grad
			
			curr_wt=curr_wt+delta_wt
			#print i,curr_error,prev_error,diff
			prev_error=curr_error
			if(abs(diff)<=0.001):
				minima_reached=True
				count=i
				break
			
		E_rms=min_err
		#print "count "+str(count)
		#print "diff: "+str(diff)
		#print "Erms: "+str(E_rms)	
		ret_val=np.array(wt,dtype='float')
		return E_rms,ret_val
	
	def train_sgd(self,reg_lambda,M,phi):

		E_rms,trained_wt=self.compute_sgd(reg_lambda,M,phi)
		return E_rms,trained_wt
	
	def get_target(self,get_data_set):
		
		complete_target=self.target[:len(self.train)+len(self.validate)+len(self.test)]
		ret_target=0
		
		if(get_data_set=="TEST"):
			st_offset=self.num_lines_train+self.num_lines_val
			ret_target=complete_target[st_offset:st_offset+self.num_lines_test]
		elif(get_data_set=="VALIDATION"):
			st_offset=self.num_lines_train
			ret_target=complete_target[st_offset:st_offset+self.num_lines_val]
		elif(get_data_set=="TRAIN"):
			st_offset=0
			ret_target=complete_target[st_offset:st_offset+self.num_lines_train]
		else:
			print "Incorrect data set selected for retrieving target values"
		
		
		ret_val=np.array(ret_target,dtype='int')
		return ret_val
	
	def get_Erms(self,trained_wt,reg_lambda,DATASET,test_phi):
		
		#Need to calculate the phi for the Test Dataset to calculate the error in prediction
		if(DATASET=="OTHERS"):
			target_test=self.get_target("TRAIN")
		else:
			target_test=self.get_target(DATASET)
		
		Ed=self.get_Ed(trained_wt,test_phi,target_test)
		Ew=self.get_Ew(trained_wt)
		sz=0
		if(DATASET=="TRAIN"):
			sz=self.num_lines_train
		elif(DATASET=="VALIDATION"):
			sz=self.num_lines_val
		elif(DATASET=="TEST"):
			sz=self.num_lines_test
		elif(DATASET=="OTHERS"):
			sz=1
		E=Ed+reg_lambda*Ew;
		Erms=np.sqrt((2*E)/sz);
		return Erms;
	
	def get_Ed(self,trained_wt,test_phi,target):
		
		sum_val=0
		for i,t in enumerate(target):
			t_phi=test_phi[i,:]
			err=np.power((t-np.dot(trained_wt.transpose(),t_phi)),2)
			sum_val=sum_val+err
		
		return 0.5*sum_val
		
	def get_Ew(self,trained_wt):
		return 0.5*np.dot(trained_wt.transpose(),trained_wt)
		
	def get_erms(self,w,reg_lambda,DATASET,t_phi):
		return self.get_Erms(w,reg_lambda,DATASET,t_phi)


if __name__ == "__main__":
	
	file1=open('result1.txt','w')
	file2=open('result2.txt','w')
	
	###################### Closed form Real Data #######################
	M=5
	dim=46
	reg_lambda=0.1
	fname='Querylevelnorm.txt'
	closed_form = LinearReg(dim)
	closed_form.parse_real_data(fname,M)
	#print "Tuning parameter"
	#tuned_params=closed_form.tune_parameters()
	print "Closed Form Linear Regression Model LeToR Data"
	#reg_lambda=tuned_params.get('RL')
	#M=tuned_params.get('M')
	
	
	Erms_train=0
	Erms_val=0
	Erms_test=0
	phi=0
	val_phi=0
	#e_rms=np.zeros((30*10,3),dtype='float')
	min_erms=99999
	min_erms_test=0
	min_erms_train=0
	min_erms_val=0
	optimal_m=0
	optimal_lambda=0
	optimal_wt=0
	#for M in range(1,31):
	for M in range(28,31):
		#for j in range(1,11):     #To reduce the computation time while grading
		for j in range(2,4):
			reg_lambda=j*0.1
			#print "M: "+str(M)+" lambda: "+str(reg_lambda)
				
			Erms_train,Erms_val,Erms_test, phi, val_phi, trained_wt=closed_form.train_rbf(reg_lambda,M)
			str_train=' '.join([str(x) for x in trained_wt])
			#print "LeToR Closed Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)
			buf="LeToR Closed Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)+" trained_wt: "+str_train
			print buf
			file1.write("%s\n" % buf)
			if(min_erms>Erms_val):     #Tuning is based on Validation
				min_erms=Erms_test
				min_erms_test=Erms_test
				min_erms_train=Erms_train
				min_erms_val=Erms_val
				optimal_m=M
				optimal_lambda=reg_lambda
				optimal_wt=str_train
	
	#print "Optimal Parameters M: "+str(optimal_m)+" lambda: "+str(optimal_lambda)+" Erms <Train,Validation,Test>: <"+str(min_erms_train)+","+str(min_erms_val)+","+str(min_erms_test)+">"
	
	buf="Optimal Parameters M: "+str(optimal_m)+" lambda: "+str(optimal_lambda)+" Erms <Train,Validation,Test>: <"+str(min_erms_train)+","+str(min_erms_val)+","+str(min_erms_test)+">"+optimal_wt
	print buf
	file1.write("%s\n" % buf)
	file1.close()
	
	####################### SGD Real Data ##############################
	#M=optimal_m
	#reg_lambda=optimal_lambda
	M=6
	reg_lambda=0.01
	Erms_train=0
	Erms_val=0
	Erms_test=0
	phi=0
	val_phi=0
	
	#e_rms=np.zeros((30*10,3),dtype='float')
	min_erms=99999
	min_erms_test=0
	min_erms_train=0
	min_erms_val=0
	optimal_m=0
	optimal_lambda=0
	optimal_wt=0
	file3=open('real_sgd.txt','w+')
	print "SGD LeToR Data"
	#for M in range(1,30,3):
	for M in range(2,5):
		#for j in range(1,11):
		for j in range(7,9):			#To reduce the computation
			reg_lambda=j*0.1
			t_phi=closed_form.basis_func("TRAIN",M)
			sgd=SGD(dim)
			sgd.parse_real_data(fname,M)
			Erms_train,w=sgd.train_sgd(reg_lambda,M,t_phi)
			#print "Wt:"
			#print w
			#reg_lambda=0.1
			str_wt=' '.join([str(x) for x in w])
			Erms_val=sgd.get_erms(w,reg_lambda,"VALIDATION",t_phi)
			#print "(Validation) Erms: "+str(Erms_val)
	
			Erms_test=sgd.get_erms(w,reg_lambda,"TEST",t_phi)
			#print "(Testset) Erms: "+ str(Erms_test)
			#buf="LeToR SGD Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" "
			buf="LeToR SGD Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)+" trained_wt: "+str_wt
			print buf
			file3.write("%s\n" % buf)
			if(min_erms>Erms_val):     #Tuning is based on Validation
				min_erms=Erms_test
				min_erms_test=Erms_test
				min_erms_train=Erms_train
				min_erms_val=Erms_val
				optimal_m=M
				optimal_lambda=reg_lambda
				optimal_wt=str_wt
	
	buf="Optimal Parameters M: "+str(optimal_m)+" lambda: "+str(optimal_lambda)+" Erms <Train,Validation,Test>: <"+str(min_erms_train)+","+str(min_erms_val)+","+str(min_erms_test)+">"+" trained_wt: "+optimal_wt
	print buf
	file3.write("%s\n" % buf)
	file3.close()
	
	
	######################## Synthetic Data ############################
	
	#M=optimal_m
	#reg_lambda=optimal_lambda
	print "Closed Form Linear Regression Model Synthetic Data"
	file_input='input.csv'
	file_output='output.csv'
	dim=10
	
	
	syn_cl_form=LinearReg(dim)
	syn_cl_form.parse_syn_data(file_input,file_output,M)
	
	Erms_train=0
	Erms_val=0
	Erms_test=0
	phi=0
	val_phi=0
	
	#e_rms=np.zeros((30*10,3),dtype='float')
	min_erms=99999
	min_erms_test=0
	min_erms_train=0
	min_erms_val=0
	optimal_m=0
	optimal_lambda=0
	optimal_wt=0
	#for M in range(1,11):
	for M in range(9,11):
		#for j in range(1,11):
		for j in range(1,3):
			reg_lambda=j*0.001
			#print "M: "+str(M)+" lambda: "+str(reg_lambda)
			Erms_train,Erms_val,Erms_test, phi, val_phi, trained_wt=syn_cl_form.train_rbf(reg_lambda,M)
			str_train=' '.join([str(x) for x in trained_wt])
			#print "Synthetic Closed Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)
			buf="Synthetic Closed Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)
			print buf
			file2.write("%s\n" % buf)
			if(min_erms>Erms_val):     #Tuning is based on Validation
				min_erms=Erms_test
				min_erms_test=Erms_test
				min_erms_train=Erms_train
				min_erms_val=Erms_val
				optimal_m=M
				optimal_lambda=reg_lambda
				optimal_wt=str_train
	
	buf="Optimal Parameters M: "+str(optimal_m)+" lambda: "+str(optimal_lambda)+" Erms <Train,Validation,Test>: <"+str(min_erms_train)+","+str(min_erms_val)+","+str(min_erms_test)+">"+" trained_wt: "+optimal_wt
	print buf
	file2.write("%s\n" % buf)
	
	file2.close()
	
	######################### SGD Over Synthetic Data ####################
	#M=optimal_m
	#reg_lambda=optimal_lambda
	M=1
	reg_lambda=0.1
	print "SGD for Synthetic Data"
	sgd_syn=SGD(dim)
	sgd_syn.parse_syn_data(file_input,file_output,M)
	Erms_train=0
	Erms_val=0
	Erms_test=0
	phi=0
	val_phi=0
	
	#e_rms=np.zeros((30*10,3),dtype='float')
	min_erms=99999
	min_erms_test=0
	min_erms_train=0
	min_erms_val=0
	optimal_m=0
	optimal_lambda=0
	optimal_wt=0
	file4=open('syn_sgd.txt','w+')
	for M in range(1,3):
		for j in range(1,3):
			reg_lambda=j*0.1
			t_phi=syn_cl_form.basis_func("TRAIN",M)
			Erms_train,w=sgd_syn.train_sgd(reg_lambda,M,t_phi)
			str_wt=' '.join([str(x) for x in w])
			#print "Wt: "+str_wt
			#print wt
			#reg_lambda=0.1
			Erms_val=sgd_syn.get_erms(w,reg_lambda,"VALIDATION",t_phi)
			#print "(Validation) Erms: "+str(Erms_val)
			Erms_test=sgd_syn.get_erms(w,reg_lambda,"TEST",t_phi)
			#print "(Testset) Erms: "+str(Erms_test)
			buf="Synthetic SGD Form M: "+str(M)+" Reg_lambda: "+str(reg_lambda)+" Erms_train: "+str(Erms_train)+" Erms_validate: "+str(Erms_val)+" Erms_test: "+str(Erms_test)+" trained_wt: "+str_wt
			print buf
			file4.write("%s\n" % buf)
			if(min_erms>Erms_val):     #Tuning is based on Validation
				min_erms=Erms_test
				min_erms_test=Erms_test
				min_erms_train=Erms_train
				min_erms_val=Erms_val
				optimal_m=M
				optimal_lambda=reg_lambda
				optimal_wt=str_wt
	
	buf="Optimal Parameters M: "+str(optimal_m)+" lambda: "+str(optimal_lambda)+" Erms <Train,Validation,Test>: <"+str(min_erms_train)+","+str(min_erms_val)+","+str(min_erms_test)+">"+" trained_wt: "+optimal_wt
	print buf
	file4.write("%s\n" % buf)
	file4.close()

	
	
	
