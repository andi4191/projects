##############################################################################
#
#      Date             Author              Description
#      07-Sept-2016     Anurag Dixit        Initial draft (UBITName: anuragdi)
#
##############################################################################

import numpy as np
#from openpyxl import load_workbook
from collections import OrderedDict
from matplotlib import pyplot as plt
import math
import xlrd
#import scipy.stats

#Data class is used to parse and store the data extracted from the seed excel file
node_list=[]
class Data(object):
	
	
	def __init__(self,fname):
		
		self.filename=fname
		self.scr = 0
		self.rsrch_ovrhd = 0
		self.admn_pay = 0
		self.tut = 0
	
	def parse_data(self):
		
		score = []
		research_overhead = []
		admin_pay_base = []
		tution = []
		'''
		book = load_workbook(self.filename)
		curr_sheet = book.get_sheet_by_name('university_data')
		if(0 == curr_sheet):
			print "Seed file missing or misspelled!!"
			return
		for i,row in enumerate(curr_sheet.iter_rows()):
			for j,col in enumerate(row):
				if(j == 2):
					score.append(col.value)
				elif(j==3):
						research_overhead.append(col.value)
				elif(j==4):
						admin_pay_base.append(col.value)
				elif(j==5):
						tution.append(col.value)
		'''
		book=xlrd.open_workbook('university data.xlsx')
		sheet=book.sheet_by_index(0)
		keys=[sheet.cell(0,col_index).value for col_index in xrange(sheet.ncols)]
		dic_l=[]
		for row_index in xrange(0, sheet.nrows):
			for col_index in xrange(sheet.ncols):
				if(col_index==2):
					score.append(sheet.cell(row_index,col_index).value)
				elif(col_index==3):
					research_overhead.append(sheet.cell(row_index,col_index).value)
				elif(col_index==4):
					admin_pay_base.append(sheet.cell(row_index,col_index).value)
				elif(col_index==5):
					tution.append(sheet.cell(row_index,col_index).value)
					
		self.scr = np.array(score[1:-1],dtype='float')
		self.rsrch_ovrhd = np.array(research_overhead[1:-1],dtype='float')
		self.admn_pay = np.array(admin_pay_base[1:-1],dtype='int')
		self.tut = np.array(tution[1:-1],dtype='int')
		
		
		
	def get_scr(self):
		return self.scr



		
	def get_rsrch_ovrhd(self):
		return self.rsrch_ovrhd
	
	def get_admn_pay(self):
		return self.admn_pay
	
	def get_tut(self):
		return self.tut
	

#Statistics class is used to perform the statistical operations over the Data class objects

class Statistics(Data):
	
	def __init__(self,filename):
		Data.__init__(self,filename)
			
	def calc_stats(self):
		self.calc_elem(Data)
	
	def get_mean(self,arr):
		return np.mean(arr)
		
	def get_variance(self,arr):
		return np.var(arr)
		
	def get_std(self,arr):
		return np.std(arr)
	
	
	def calc_elem(self,dat):
		
		mu1=np.mean(self.scr)
		mu2=np.mean(self.rsrch_ovrhd)
		mu3=np.mean(self.admn_pay)
		mu4=np.mean(self.tut)
		
		var1=np.var(self.scr)
		var2=np.var(self.rsrch_ovrhd)
		var3=np.var(self.admn_pay)
		var4=np.var(self.tut)
		
		sigma1=np.std(self.scr)
		sigma2=np.std(self.rsrch_ovrhd)
		sigma3=np.std(self.admn_pay)
		sigma4=np.std(self.tut)
		ubit_name="anuragdi"
		pn="50208152"
		print "UBitName = %s" %ubit_name
		print "personNumber = %s" %pn
		
		print "mu1 = %0.2f" % mu1
		print "mu2 = %0.2f" % mu2
		print "mu3 = %0.2f" % mu3
		print "mu4 = %0.2f" % mu4
		
		print "var1 = %0.2f" % var1
		print "var2 = %0.2f" % var2 
		print "var3 = %0.2f" % var3
		print "var4 = %0.2f" % var4
		
		print "sigma1 = %0.2f" % sigma1
		print "sigma2 = %0.2f" % sigma2 
		print "sigma3 = %0.2f" % sigma3
		print "sigma4 = %0.2f" % sigma4
	
	
	def get_cov(self,a,b):
		return np.cov(a,b)
	
	def get_corrcoef(self,a,b):
		return np.corrcoef(a,b)
		
	def graph_plot(self,arr,labelx,labely):
		
		k=0
		fig = plt.figure()
		for i in range(0,2):
			for j in range(0,2):
				ax[k]=fig.add_subplot(2,i,j)
				ax[idx].plot(arr[k])
				k=k+1
		
		plt.show()
		
		
	def get_cov_cor(self):
		
		np.set_printoptions(precision=2)
		mat = np.vstack([self.scr, self.rsrch_ovrhd,self.admn_pay,self.tut])
				
		print "Covariance Matrix"
		covarianceMat = np.cov(mat)
		print covarianceMat
		print "Correlation Matrix"
		correlationMat = np.corrcoef(mat)
		print correlationMat
		
		#comb = OrderedDict([('S' , self.scr), ('R', self.rsrch_ovrhd), ('B' , self.admn_pay), ( 'T' , self.tut)])
		comb = OrderedDict([('Score' , self.scr), ('Research_Ovrhd', self.rsrch_ovrhd), ('Base_Pay' , self.admn_pay), ( 'Tution' , self.tut)])
		comb_keys = comb.keys()
		cov_list = []
		xlab = []
		ylab = []
		k=0
		
		for i in range(0,4):
			for j in range(i+1,4):
				
				cov_list.append(covarianceMat[i][j])
				xlab.append(comb_keys[i]+","+comb_keys[j])
				k+=1
		self.plotting(cov_list,xlab,'blue',"Covariance Between parameters")

		# For correlation Graph
		corr_list = []
		xlab = []
		k=0
		min_val = 99999999
		min_index=0
		max_index = 0
		max_val = -99999999
		print "Correlation Graph in separate windows"
		
		for i in range(0,4):
			for j in range(i+1,4):
				
				#seed1=comb[comb_keys[i]]
				#seed2=comb[comb_keys[j]]
				corr_list.append(correlationMat[i][j])
				xlab.append(comb_keys[i]+","+comb_keys[j])
				
				#Finding least correlated pair
				if(correlationMat[i][j]<min_val):
					min_val = correlationMat[i][j]
					min_index = k
					
				#Finding most correlated pair
				if(correlationMat[i][j]>max_val):
					max_val = correlationMat[i][j]
					max_index = k
				
				k+=1
		self.plotting(corr_list,xlab,'green',"Correlation Between parameters")
		self.plott(cov_list,corr_list,xlab)
		self.plotti(cov_list,corr_list,xlab)
		self.plot_scatter(self.scr,self.rsrch_ovrhd,self.admn_pay,self.tut)
		np.set_printoptions(precision=2)
		
		print "Least Correlated pair %s with value %0.2f" %(xlab[min_index], min_val)
		print "Most Correlated pair %s with value %0.2f" % (xlab[max_index], max_val)
		
		loglikelihood=0		
		for i in range(0,4):
			
			loglikelihood+=self.loglikeihood_calc(comb[comb_keys[i]],self.get_mean(comb[comb_keys[i]]),self.get_variance(comb[comb_keys[i]]),comb_keys[i])
		
		
		print "logLikelihood: %0.2f" %(loglikelihood) 
		
		self.bayesian_ntwork(corr_list,xlab)
	
	def plotting(self, y_list,xlab,colr,title):
				
		fig = plt.figure()
		ax= fig.add_subplot(111)
		ind = np.arange(len(y_list))
		wid = 0.35
		#print xlab
		rec = ax.bar(ind,y_list,wid,color=colr)
		ax.set_xlim(-wid,len(ind)+wid)
		ax.set_title(title)
		xtickm=[i for i in xlab]
		ax.set_xticks(ind+wid)
		xtickname = ax.set_xticklabels(xtickm)
		plt.setp(xtickname,rotation=15, fontsize=10)
		plt.show()
	
	def plott(self,cov_list,corr_list,xlab):
		grp = len(corr_list)
		fix,ax=plt.subplots()
		ind = np.arange(grp)
		bar_width = 0.35
		opacity=0.8
		
		rec1 = plt.bar(ind,cov_list, bar_width, alpha=opacity,color='b',
		label='Covariance')
		rec2 = plt.bar(ind+bar_width,corr_list, bar_width, alpha=opacity,color='g',
		label='Correlation')
		
		plt.xlabel('Parameters')
		plt.ylabel('Value')
		plt.title('Relation between parameters')
		plt.xticks(ind+bar_width,xlab)
		xtickm=[i for i in xlab]
		xtickname = ax.set_xticklabels(xtickm)
		plt.setp(xtickname,rotation=15, fontsize=10)
		plt.legend()
		plt.tight_layout()
		plt.show()
	
	def plot_scatter(self,scr,rsrch_ovrhd,admn_pay,tut):
		val=[scr,rsrch_ovrhd,admn_pay,tut]
		title=['CS Score','Research Overhead','Admission Base Pay','Tution']
		
		fig,axes=plt.subplots(nrows=4,ncols=4,figsize=(9,9))
		plt.suptitle('Covariance between parameters')
		for num_i,i in enumerate(title):
			
			for num_j,j in enumerate(title):
				if num_i==len(title)-1:
					axes[num_i,num_j].set_xlabel(title[num_j])				
				if num_j==0:
					axes[num_i][num_j].set_ylabel(title[num_i])
				axes[num_i,num_j].plot(val[num_i],val[num_j],'o')
				for tick in axes[num_i][num_j].get_xticklabels():
					tick.set_rotation(25)
				
		
		
		plt.tight_layout()
		
		plt.show()
		
	def plotti(self,cov_list,corr_list,xlab):
		
		#histplot(cov_list)
		#histplot(corr_list)
		fig = plt.figure()
		ax = plt.axes()
		x_dat = range(0,len(xlab))
		plt.plot(x_dat,cov_list,label='covariance')
		plt.plot(x_dat,corr_list,label = 'correlation')
		#ax.plot(len(xlab),cov_list,'b',label='covariance')
		#ax.plot(len(xlab),corr_list,'g',label='correlation')
		xtickm=[i for i in xlab]
		xtickname = ax.set_xticklabels(xtickm)
		plt.setp(xtickname,rotation=15, fontsize=10)
		plt.legend()
		plt.show()
	
	
	def loglikeihood_calc(self,x,mean,var,param):
		
		sum_log=0
		for i in x:
			coeff=1/(math.sqrt(2*math.pi*var))
			expon=0-0.5*(pow((i-mean),2)/var)
			sum_log+=np.log(coeff*np.exp(expon))
			#just to verify for debugging
			#spdf+=scipy.stats.norm.logpdf(i,mean,var)	
			
		return sum_log
		
	
	def max_likelihood(self,bngraph,node):
		
		pass
		
	def map_value(map_dat,x):
		return map_dat[x]
	
	def get_prod(self,a,b):
		sum_=0
		for i in range(0,len(a)):
			sum_+=a[i]*b[i]
		
		return sum_
		
	def get_lglkhood_conditional_probabilities(self,a,b):
		
		map_dat=[self.scr,self.rsrch_ovrhd,self.admn_pay,self.tut]
		#scr=0; rsrch_ovrhd=1; admn_pay=2; tution=3
		
		variable_x=[]
		variable_x=b
		k=len(variable_x)
		A=np.empty([k+1,k+1],dtype=float)
		beta=np.empty([k+1,1],dtype=float)
		y=np.empty([k+1,1],dtype=float)
		unit_vec=np.ones([49,1],dtype=float)
		
		y[0]=self.get_prod(unit_vec,map_dat[a])
		for idx,i in enumerate(variable_x):
			y[idx+1]=self.get_prod(map_dat[a],map_dat[i])
			
		for idx,i in enumerate(variable_x):
			vec_i=map_dat[i]
			A[idx][0]=self.get_prod(vec_i,unit_vec)
			A[0][idx]=self.get_prod(vec_i,unit_vec)
		
		for idx,i in enumerate(variable_x):
			for idy,j in enumerate(variable_x):
				vec_i=map_dat[i]
				vec_j=map_dat[j]
				A[idx+1][idy+1]=self.get_prod(vec_i,vec_j)
		
		
		try:
			beta=np.dot(np.linalg.inv(A),y)
		except np.linalg.linalg.LinAlgError as err:
			if 'Singular matrix' in err.message:
				#print"Singular matrix" 
				pass
		
		lglikelihood=0
		
		for idx,i in enumerate(variable_x):
			vec_x=map_dat[i]
			sum_=self.get_prod(beta[0],unit_vec)
			count=0
			var=0
			
			for j in range(0,len(vec_x)):
				
				vec_y=map_dat[a]
				sum_+=pow((beta[idx+1]*vec_x[j]-vec_y[j]),2)
				count+=1
			
			var=sum_/count;
			lglikelihood+=self.calc_likelihood(var, sum_,count)
		
		return lglikelihood
		
	def calc_likelihood(self,var,sum_,count):
		
		const_val=0-0.5*np.log(2*math.pi*var)
		res=(count*const_val)-((0.5*sum_)/var)
		return res
	
	def det_dag(self,node_list,parent,child,max_loglikelihood,visited):
		curr_val=self.get_lglkhood_conditional_probabilities(child,parent)
		if(len(visited)==3):
			return max_loglikelihood
		if(curr_val>max_loglikelihood):
			max_loglikelihood=curr_val
			node_list.append(child)
			node_list.append('next')
			visited.append(child)
		
		else:
			return (self.det_dag(node_list,parent,(child+1)%4,max_loglikelihood,visited))
	
	def get_independent_likelihood(self,x,mean,var,map_dat,variable_x):
		z=map_dat[variable_x]
		val=0
		for i in z:
			val+=scipy.stats.norm.logpdf(i,mean,var)
		
		return val
	
	def get_other_prob(self,child,parent):
		others=[]
		arr=[self.scr,self.rsrch_ovrhd,self.admn_pay,self.tut]
		for i in range(0,4):
			if(i in parent):
				others.append(i)
			elif(i == child):
				continue
			else:
				others.append(i)
		
		val=0
		for i in others:
			val+=self.loglikeihood_calc(arr[i],np.mean(arr[i]),np.var(arr[i]),0)
		return val
	
	def const_graph(self):
		
		#Scr=0; RO=1; BP=2; Tut:3
		arr=[self.scr,self.rsrch_ovrhd,self.admn_pay,self.tut]
		
		bngraph=np.zeros([4,4],dtype=int)
		visited=[]
		parent=[0]
		child=1
		
		par=[]
				
		my_dict={'0':99,'1':99,'2':99,'3':99}
		
		max_likelihood=-9999
		parent=[]
		for i in range(0,4):
			child=i
			parent=[]
			for j in range(0,4):
				if(i==j):
					continue
				else:
					parent.append(j)
					curr_val=self.get_lglkhood_conditional_probabilities(child,parent)
					curr_val+=self.get_other_prob(child,parent)
					#print "child :%d val:%0.2f"%(child,curr_val)
					#print "parents"
					#print parent
					if(curr_val>max_likelihood):# and len(parent)==3):
						max_likelihood=curr_val
						par=parent
						final_child=child
						dic={str(final_child):par}
						my_dict.update(dic)
						for k in par:
							bngraph[k][child]=1
		
		'''
		_key=my_dict.keys()
		for i,val in enumerate(_key):
			#print i, val
			if(my_dict[val]==99):
				bngraph[i][i]=1
		'''
		#print my_dict
		print "BNgraph" 
		print bngraph
		print "BNlogLikelihood %0.2f" %(max_likelihood)
						
				
		
		
	def bayesian_ntwork(self,corr_list,xlab):
		self.const_graph()
		pass
		
		
	
	
if __name__ == "__main__":
	st = Statistics('university data.xlsx')
	st.parse_data()
	st.calc_stats()
	st.get_cov_cor()

