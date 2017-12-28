import sys
import csv
import numpy as np

data_path = sys.argv[1] #'/home/lab249/ML2016/final/data/'
type_path = data_path + 'training_attack_types.txt'
train_path = 'newtrain'
test_path = data_path + 'test.in'

attack_type = []
attack_label = []

protocal_type = []
service = []
flag = []

N_train = 0
N_test = 0
train_X = []
train_y = []
test_X = []

attack_type.append('normal.')
attack_label.append('0')
file = open(type_path, 'r')
for line in file:
	line = line.strip().split()
	attack_type.append(line[0] + '.')
	if line[1] == 'dos':
		attack_label.append('1')
	elif line[1] == 'u2r':
		attack_label.append('2')
	elif line[1] == 'r2l':
		attack_label.append('3')
	elif line[1] == 'probe':
		attack_label.append('4')
	else:
		print 'error'
file.close()

file = open(train_path, 'r')
for line in csv.reader(file):
	try:
		protocal_type.index(line[1])
	except ValueError:
		protocal_type.append(line[1])

	try:
		service.index(line[2])
	except ValueError:
		service.append(line[2])

	try:
		flag.index(line[3])
	except ValueError:
		flag.append(line[3])
file.close()
'''
print 'protocal_type:', len(protocal_type)
print protocal_type
print 'service:', len(service)
print service
print 'flag:', len(flag)
print flag
'''
file = open(train_path, 'r')
for line in csv.reader(file):
	temp_x = []
	temp_x.append(line[0])

	p_vec = np.zeros(len(protocal_type))
	p_idx = protocal_type.index(line[1])
	p_vec[p_idx] += 1
	for i in range(len(protocal_type)):
		temp_x.append(str(int(p_vec[i])))

	s_vec = np.zeros(len(service))
	s_idx = service.index(line[2])
	s_vec[s_idx] += 1
	for i in range(len(service)):
		temp_x.append(str(int(s_vec[i])))

	f_vec = np.zeros(len(flag))
	f_idx = flag.index(line[3])
	f_vec[f_idx] += 1
	for i in range(len(flag)):
		temp_x.append(str(int(f_vec[i])))
	
	for i in range(4, 41):
		temp_x.append(line[i])

	label_idx = attack_type.index(line[-1])
	temp_y = attack_label[label_idx]

	N_train += 1
	train_X.append(temp_x)
	train_y.append(temp_y)
file.close()

file = open('train_X.csv', 'w')
for n in range(N_train):
	csv.writer(file).writerow(train_X[n])
file.close()

file = open('train_y.csv', 'w')
for n in range(N_train):
	csv.writer(file).writerow(train_y[n])
file.close()

file = open(test_path, 'r')
for line in csv.reader(file):
	temp_x = []
	temp_x.append(line[0])

	p_vec = np.zeros(len(protocal_type))
	try:
		p_idx = protocal_type.index(line[1])
		p_vec[p_idx] += 1
	except ValueError:
		print N_test, line[1]
	for i in range(len(protocal_type)):
		temp_x.append(str(int(p_vec[i])))

	s_vec = np.zeros(len(service))
	try:
		s_idx = service.index(line[2])
		s_vec[s_idx] += 1
	except ValueError:
		print N_test, line[2]
	for i in range(len(service)):
		temp_x.append(str(int(s_vec[i])))

	f_vec = np.zeros(len(flag))
	try:
		f_idx = flag.index(line[3])
		f_vec[f_idx] += 1
	except ValueError:
		print N_test, line[3]
	for i in range(len(flag)):
		temp_x.append(str(int(f_vec[i])))
	
	for i in range(4, 41):
		temp_x.append(line[i])

	N_test += 1
	test_X.append(temp_x)
file.close()

file = open('test_X.csv', 'w')
for n in range(N_test):
	csv.writer(file).writerow(test_X[n])
file.close()
