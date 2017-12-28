from sklearn import linear_model, svm, metrics, datasets, model_selection, tree, linear_model, neural_network
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

#Regression part
def regression(X_train, X_test, y_train, y_test, test_data):
	logr = linear_model.LogisticRegression(penalty='l2',
		solver='liblinear',
		multi_class='ovr',
		verbose=0,
		n_jobs=1)
	logr.fit(X_train, y_train)

	print(logr.score(X_train, y_train))
	print(logr.score(X_test, y_test))

	prediction = logr.predict(test_data)
	return prediction

#Decision Tree part
def decision_tree(X_train, X_test, y_train, y_test, test_data):
	dct = tree.DecisionTreeClassifier(criterion='gini',
		splitter= 'best',
		max_depth=None,
		min_samples_split=3,
		max_features=None,
		max_leaf_nodes=None,
		min_impurity_decrease=0.0)
	dct.fit(X_train, y_train)

	print(dct.score(X_test, y_test))
	print(dct.score(X_train, y_train))

	prediction = dct.predict(test_data)
	return prediction

#SVM part
def svm_(X_train, X_test, y_train, y_test, test_data):
	svc_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='crammer_singer', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
	
	svc_model.fit(Xtrain, ytrain)
	print(svc_model.score(Xtrain, ytrain))
	print(svc_model.score(Xtest, ytest))
	
	prediction = svc_model.predict(test_data)
	return prediction

#Neural Network part
def neural_nwk(X_train, X_test, y_train, y_test, test_data):
	mlp = neural_network.MLPClassifier(activation='identity', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(25,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

	#data scaling
	scalar = StandardScaler()
	scalar.fit(Xtrain)
	Xtrain2 = scalar.transform(Xtrain)
	Xtest2 = scalar.transform(Xtest)
	
	#first fit
	mlp.fit(X_train, y_train)
	print(mlp.score(X_train, y_train))
	print(mlp.score(X_test, y_test))

	#fit the transformed datasets
	mlp.fit(Xtrain2, ytrain)
	print(mlp.score(Xtrain2, ytrain))
	print(mlp.score(Xtest2, ytest))
	
	prediction = mlp.predict(test_data)
	return prediction

#determine which method to call
def call_method_type_from_cmd(type,X_train, X_test, y_train, y_test, td):
	if   type == 'R':
		method = regression(X_train, X_test, y_train, y_test, td)
	elif type == 'D':
		method = decision_tree(X_train, X_test, y_train, y_test, td)
	elif type == 'S':
		method = svm_(X_train, X_test, y_train, y_test, td)
	elif type == 'N':
		method = neural_nwk(X_train, X_test, y_train, y_test, td)
	else:
		method = None
		print("wrong argument!")
	return method

#read the real train csv files
def read_train_csv_file(filename,header=-1):
	df = pd.read_csv(filename)
	X,y = df.iloc[:,:-1],df.iloc[:,-1]
	print(X.shape)
	print(y.shape)
	return X,y

#read train csv files using pandas for tuning the nodel
def read_train_csv_file_and_divide_train_test(filename,header=-1):
	df = pd.read_csv(filename)
	X,y = df.iloc[:,:-1],df.iloc[:,-1]
	Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X,y, test_size = 0.05)
	Xtrain, Xtest, ytrain, ytest = Xtrain.values, Xtest.values, ytrain.values, ytest.values
	return Xtrain, Xtest, ytrain, ytest

#for datascience final project
def auto_read_train_csv_file():
	X = pd.read_csv("train_X.csv")
	y = pd.read_csv("train_y.csv")
	Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X,y, test_size = 0.05)
	Xtrain, Xtest, ytrain, ytest = Xtrain.values, Xtest.values, ytrain.values, ytest.values
	print(Xtrain.shape)
	print(ytrain.shape)
	print(Xtest.shape)
	print(ytest.shape)

	return Xtrain, Xtest, ytrain, ytest

def auto_read_test_csv_file():
	 test_data = pd.read_csv("test_X.csv")
	 test_data  = test_data.values
	 print("shape of test_data" , test_data.shape)
	 return test_data

#read test csv files using pnadas
def read_test_csv_file(filename):
	test_df = pd.read_csv(filename,header=-1)
	return test_df

#write to "predict.csv" file
def write_predict_csv_file(pred):
	pred = pd.DataFrame(pred, columns=['predictions']).to_csv('prediction.csv')
	return True

#read arguments from cmd inputs
def readall_arguments_from_cmd():	
	for arg in sys.argv:
		arguments.append(arg)

arguments = []
readall_arguments_from_cmd()

###########FOR HW2###########
#Xtrain, Xtest, ytrain, ytest = read_train_csv_file_and_divide_train_test(arguments[2])
#test_data = read_test_csv_file(arguments[3])

#Xtrain, ytrain = read_train_csv_file(arguments[2])

###########FOR FINAL PROJECT###########
Xtrain, Xtest, ytrain, ytest = auto_read_train_csv_file()
test_data = auto_read_test_csv_file()

prediction = call_method_type_from_cmd(arguments[1], Xtrain, Xtest, ytrain, ytest, test_data)
write_predict_csv_file(prediction)