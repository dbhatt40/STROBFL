#########################
# Purpose: Help with file input/output
########################

import global_vars as gv
import numpy as np

from .census_utils import data_census
from .gas_sensor_utils import data_uci_sensor


def file_write(write_dict, purpose='global_eval_loss'):
	f = open(gv.output_dir_name + gv.output_file_name +
	         '_' + purpose + '.txt', 'a')
	if write_dict['t'] == 1:
		d_count = 1
		for k, v in iter(write_dict.items()):
			if d_count < len(write_dict):
				f.write(k + ',')
			else:
				f.write(k + '\n')
			d_count += 1
		d_count = 1
		for k, v in iter(write_dict.items()):
			if d_count < len(write_dict):
				f.write(str(v) + ',')
			else:
				f.write(str(v) + '\n')
			d_count += 1
	elif write_dict['t'] != 1:
		d_count = 1
		for k, v in iter(write_dict.items()):
			if d_count < len(write_dict):
				f.write(str(v) + ',')
			else:
				f.write(str(v) + '\n')
			d_count += 1
	f.close()
    

def file_writetime(write_dict, purpose='training_time'):
		f = open(gv.output_dir_name + gv.output_file_name +
		         '_' + purpose + '.txt', 'a')
		if write_dict['t'] == 1:
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(k + ',')
				else:
					f.write(k + '\n')
				d_count += 1
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1
		elif write_dict['t'] != 1:
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1
		f.close()

def file_writemaindata(write_dict, purpose='main_training_data'):
		filename = gv.output_dir_name + gv.output_file_name + '_' + purpose + '.txt'
		f = open(filename,'a')
		if write_dict['t'] == 20:			
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(k + ',')
				else:
					f.write(k + '\n')
				d_count += 1
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1	
		elif write_dict['t'] != 20:			
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1
		f.close()

def file_writemetricsdata(write_dict, purpose='metrics_data'):
		filename = gv.output_dir_name + gv.output_file_name + '_' + purpose + '.txt'
		f = open(filename,'a')
		if write_dict['t'] == 1:			
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(k + ',')
				else:
					f.write(k + '\n')
				d_count += 1
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1	
		elif write_dict['t'] != 1:		
			d_count = 1
			for k, v in iter(write_dict.items()):
				if d_count < len(write_dict):
					f.write(str(v) + ',')
				else:
					f.write(str(v) + '\n')
				d_count += 1
		f.close()

def write_matrix(mat, purpose):
    filename = gv.output_dir_name + gv.output_file_name + '_' + purpose 
    f = open(filename,'a')  
    np.save(filename,mat)
    f.close()
	

def data_setup():
	args = gv.args
	if args.dataset == 'census':
		X_train, Y_train, X_test, Y_test = data_census()
		Y_test_uncat = np.argmax(Y_test, axis=1)
		# print Y_test
		# print Y_test_uncat
		print('Loaded Census data')
	elif args.dataset == 'uci-sensor':
			X_train, Y_train, X_test, Y_test = data_uci_sensor()
			Y_test_uncat = Y_test
			# print Y_test
			# print Y_test_uncat
			print('Loaded UCI sensor data')

	return X_train, Y_train, X_test, Y_test, Y_test_uncat

