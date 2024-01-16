import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numpy import random
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


def mydist(x,y):
	z=abs(x-y)
	xx=np.sort(z)
	###print(min(z))
	return (xx[0]+xx[1])


############################## Put filename on Line #35 
##############################

dataset = pd.read_csv("pathname", header=None)
shape=dataset.shape
###num_class=11
num_label=1

print("Shape ={}\n". 
format(shape))

X = dataset.iloc[:, :-num_label].values
y = dataset.iloc[:, -num_label:].values
###print(X.max(axis=0))
dd=X.min(axis=0)

ee=X.max(axis=0)

ee[ee==0]=1

X_normed =( X-X.min(axis=0)) / (ee-X.min(axis=0))
X=X_normed

###print(dd.shape)
###print(ee.shape)
###print(X.max(axis=0))
###print(X.max(axis=0))

shape=X.shape
###print(shape)


###from sklearn.model_selection import train_test_split
###X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)




ss=range(X.shape[0])

pp1=np.random.choice(X.shape[0],int(np.ceil(X.shape[0]/2)),replace=False)
pp2=np.setdiff1d(ss, pp1)

X_train=X[pp1,:]
y_train=y[pp1,:]
X_test=X[pp2,:]
y_test=y[pp2,:]


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
pred_result=neigh.predict(X_test)
pred_result=np.reshape(pred_result,(1, pred_result.size))
pred_result=pred_result.T

###result=sum(pred_result==y_test)/y_test.shape[0]

###result=f1_score(y_test,pred_result,average='weighted')
	
result=accuracy_score(y_test,pred_result)
	

print(result)
print(y_test.shape)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_test)


	


num_it=1
max_it=1

num_features=16


f_score = open("result_privacy_score.txt",'a+')
f_matching=open("result_privacy_matching.txt",'a+')


while num_it<=max_it:



	f_score.write(str(result))
	f_score.write(',')
	
	p=max((np.ceil(math.log(X.shape[0],2)))-X.shape[1],2)
	p=int(p)
	###print(p)
	###print(p)
	n=X.shape[1]
	###print(X.shape[0])
	###print(X.shape[1])
	###print(p)
	###print(p)

	
	num_it=num_it+1
	
	New_features=np.zeros((X.shape[0],1),dtype='float') 


	for ii in range(num_features):

		XX=X
		
		perm=np.random.permutation(X.shape[1])
		##print(perm)
		XX=XX[:,perm]
		###print(XX.shape)
		
		
		from hilbertcurve.hilbertcurve import HilbertCurve	
		hilbert_curve = HilbertCurve(p, n)
		XX=np.floor(XX*(pow(2,p)-1))
		###X=X*p
		###X_test=X_test*hilbert_curve.max_x+1
		###num_points = 10_000
		##points = np.random.randint(low=0,high=hilbert_curve.max_x + 1,size=(num_points, hilbert_curve.n))
		points=XX
		##print(points.shape)

		distances = hilbert_curve.distances_from_points(points)
		distances=np.array(distances)
		if (hilbert_curve.max_x+1)<64:
			distances=distances/np.power(2,hilbert_curve.max_x+1)
		else:
			distances=distances/np.power(2,(hilbert_curve.max_x+1)/2)
				
		distances = np.reshape(distances,(1, distances.size))
		##print(New_features.shape)
		###print(distances.shape)
		New_features=np.hstack((New_features,distances.T))
		
		
		###print(hilbert_curve.max_x+1)
		###print(distances)   
	
	###print(New_features.shape)
	New_features=New_features[:,1:]
	New_features=New_features/(pow(2,p)-1)
	###print(New_features.min(0))
	###print(New_features.max(0))
	
	###from sklearn.model_selection import train_test_split
	###X_train, X_test, y_train, y_test = train_test_split(New_features, y, test_size=0.50)
	
	
	X_train=New_features[pp1,:]
	y_train=y[pp1,:]
	X_test=New_features[pp2,:]
	y_test=y[pp2,:]
	
	
	
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(X_train, y_train)
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T

	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	result=accuracy_score(y_test,pred_result)
	print(result)
	
	f_score.write(str(result))
	f_score.write(',')
	
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices.shape[0]
	
	print(matching)
	
	f_matching.write(str(matching))
	f_matching.write(',')
	
	
	
	
	neigh = KNeighborsClassifier(n_neighbors=5,metric=mydist)
	neigh.fit(X_train, y_train)
	
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T

	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	result=accuracy_score(y_test,pred_result)
	print(result)
	
	f_score.write(str(result))
	f_score.write(',')
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric=mydist).fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices1.shape[0]
	
	print(matching)
	
	f_matching.write(str(matching))
	f_matching.write(',')
	
	##print(y_test.shape)
	
	###print(X_train.mean(0))
	###print(X_test.mean(0))
	
	pp=np.random.choice(num_features,int(num_features/2),replace=False)
	X_train=X_train[:,pp]
	X_test=X_test[:,pp]
	
	
	
	
	
	print(X_train.shape)
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(X_train, y_train)
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T
	result=accuracy_score(y_test,pred_result)
	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	print(result)
	
	f_score.write(str(result))
	f_score.write(',')
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices1.shape[0]
	
	print(matching)
	
	f_matching.write(str(matching))
	f_matching.write(',')
	
	
	neigh = KNeighborsClassifier(n_neighbors=5,metric=mydist)
	neigh.fit(X_train, y_train)
	
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T

	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	result=accuracy_score(y_test,pred_result)
	print(result)
	
	f_score.write(str(result))
	f_score.write(',')
	
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric=mydist).fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices1.shape[0]
	
	print(matching)
	
	f_matching.write(str(matching))
	f_matching.write(',')


	pp=np.random.choice(int(num_features/2),int(num_features/4),replace=False)
	X_train=X_train[:,pp]
	X_test=X_test[:,pp]
	print(X_train.shape)
	
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(X_train, y_train)
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T
	result=accuracy_score(y_test,pred_result)
	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	print(result)
	
	f_score.write(str(result))
	f_score.write(',')
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices1.shape[0]
	
	f_matching.write(str(matching))
	f_matching.write(',')
	
	print(matching)
	
	neigh = KNeighborsClassifier(n_neighbors=5,metric=mydist)
	neigh.fit(X_train, y_train)
	
	pred_result=neigh.predict(X_test)
	pred_result=np.reshape(pred_result,(1, pred_result.size))
	pred_result=pred_result.T

	###result=sum(pred_result==y_test)/y_test.shape[0]
	###result=f1_score(y_test,pred_result,average='weighted')
	
	result=accuracy_score(y_test,pred_result)
	print(result)
	
	f_score.write(str(result))
	f_score.write('\n')
	
	
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric=mydist).fit(X_train)
	distances, indices1 = nbrs.kneighbors(X_test)
	matching=np.sum(indices==indices1)/indices1.shape[0]
	
	print(matching)
	
	f_matching.write(str(matching))
	f_matching.write('\n')


f_matching.write('\n')
f_score.write('\n')


          
