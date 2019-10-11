import tensorflow as tf 
import numpy as np 

from datetime import datetime
import os, sys

import itr_matcher

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_type', help='the type of model to use: I3D')

parser.add_argument('dataset_dir', help='the directory where the dataset is located')
parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

parser.add_argument('pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')

#feature pruning command line args
parser.add_argument('--dataset_id', nargs='?', type=int, default=4, help='the dataset_id used by the csv file')
parser.add_argument('--feature_retain_count', nargs='?', type=int, default=-1, help='the number of features to remove')

parser.add_argument('--gpu_memory', nargs='?', type=float, default=0.5, help='how much of the GPU should the process consume')
parser.add_argument('--gpu', default="0", help='gpu to run on')

FLAGS = parser.parse_args()

IAD_DATA_PATH = os.path.join(FLAGS.dataset_dir, 'iad')
DATASET_ID_PATH = os.path.join('itr', "dataset_"+str(25*FLAGS.dataset_id))
ITR_DATA_PATH = os.path.join(FLAGS.dataset_dir, DATASET_ID_PATH)


sys.path.append("../../IAD-Generator/iad-generation/")
from feature_rank_utils import get_top_n_feature_indexes
from csv_utils import read_csv

input_shape_i3d = [(64, FLAGS.pad_length/2), (192, FLAGS.pad_length/2), (480, FLAGS.pad_length/2), (832, FLAGS.pad_length/4), (1024, FLAGS.pad_length/8)]
input_shape = input_shape_i3d

'''
#0 - empty
#1 - br
#2 - bl
#3 - b
#4 - tr
#5 - r
#6 - tr, bl
#7 - r, bl
#8 - tl
#9 - tl, br
#10 - l
#11 - l, br
#12 - t
#13 - r, tl
#14 - l, tr
#15 - full

CORE_FILTERS_13 = {
		'meets': np.array([[9]]),
		'metBy': np.array([[6]]),
		'starts': np.array([[5,11]]),
		'startedBy': np.array([[5,14]]),
		'finishes': np.array([[7,10]]),
		'finishedBy': np.array([[13,10]]),
		'overlaps': np.array([[13,11]]),
		'overlapedBy': np.array([[7,14]]),
		'during': np.array([[7,11]]),
		'contains': np.array([[13,14]]),
		'before': np.array([[8,1]]),
		'after': np.array([[2,4]]),
		'equals': np.array([[5,10]])
	}
'''

write_file,run_code = 0, 0

############################
# Pairwise Combinations
############################

def pairwise_gather(input_var):
	'''
	Generates a pairwise combination of all the rows in the input tensor
	'''
	num_features = input_var.get_shape()[0]

	# get all unique pairwise combinations of two features
	indices = []
	for i in range(num_features):
		for j in range(i, num_features):
			if(i != j):
				indices.append([[i], [j]])

	return tf.gather_nd(input_var, indices, name="pairwise_gather")

############################
# Convolution
############################

def permute_filter_combinations(shared_array, array, cur_depth):
	'''
	Recursively get all possible filter combinations of -1 and 1 for a filter shape
	'''
	if(cur_depth == len(array)):
		shared_array.append(array)
	else:
		for value in [-1,1]:
			array[cur_depth] = value
			permute_filter_combinations(shared_array, array[:], cur_depth+1)

def get_variables(filter_width):
	'''
	Generate the fixed filters used in the convolution
	'''
	filters = []
	permute_filter_combinations(filters, [1]*(2*filter_width), 0)
	filters = np.array(filters).reshape((-1, 2, filter_width))

	# modify the shape of the filters to allow convolution by tensorflow
	filters = np.expand_dims(filters, axis = 0)
	filters = np.transpose(filters, (2,3, 0, 1))
	filters = tf.cast(tf.constant(filters),tf.float32)
	return filters

def generate_pairwise_projections(input_ph, filter_width=2):
	'''
	Generate the projection of filters for each of the pairwise combinations
	of rows in the IAD. This involves two steps: 1) the pairwise seprataion 
	of features and 2) the convolution of each pairwise combination with a 
	specific set of fixed filters (known as moments). The best moment at each
	time instance is then projected into a 1-D array for the pairwise combination
		- placeholders: dict, placeholders for the convolution 
				(see get_placeholders function)
		- filter_width: int, default (2) the width of the fixed filters
	'''

	# get pairwise combination of IAD features
	pw = pairwise_gather(input_ph)
	
	# pad the beginning and ending of each IAD with -1 values to ensure that 
	# specific core moments can be found
	pw = tf.pad(pw, tf.constant([[0,0], [0,0], [1,1]]), "CONSTANT", constant_values = -1)
	
	#reshape combinations for subsequent operations
	pw = tf.expand_dims(pw, axis = 0)
	pw = tf.transpose(pw, perm=[1,2,3,0])

	#convolve
	variables = get_variables(filter_width)
	conv0 = tf.nn.conv2d(pw, variables, strides=[1,1,1,1], padding='VALID', name="conv2d_op")
	conv0 = tf.squeeze(conv0, name="squeeze_op", axis=1)

	#collapse data
	return tf.argmax(conv0, axis = 2)

############################
# Collapse and Count
############################

def extract_itrs(projections):
	#global write_file, run_code
	'''
	Identify the IADs present in the pairwise projections generated by the 
	"generate_pairwise_projections" function. This is done by manipulating 
	the data so that it can be passed into a function that runs in C++
		- projections: tensor, a tensor containing the projections for each 
				pairwise combination of features in the IAD
	'''

	# get the properties of the input
	num_pw_combinations = projections.shape[0]
	num_time_instances = projections.shape[1]
	num_itrs = 13

	# write information to file to be read in C++
	projections = projections.astype(np.int64)

	# prepare output array
	itr_counts = [0]*(num_pw_combinations * num_itrs)

	# extract ITRs using C++
	itr_matcher.thread(projections, num_pw_combinations, num_time_instances, itr_counts)

	# reshape the resultant array into a reasonable format
	return np.array(itr_counts).reshape((num_pw_combinations, num_itrs))

############################
# Main
############################

def extract_itrs_by_layer(csv_contents, layer, pruning_keep_indexes=None):
	###############################
	# Extract ITRs
	###############################

	# get the pairwise projection (one dimensional representations of the data)
	#print (data.shape)
	ph = tf.placeholder(tf.float32, shape=(input_shape[layer][0],FLAGS.pad_length),name="input_ph")
	itr_extractor = generate_pairwise_projections(ph)

	# prevent TF from consuming entire GPU
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)

	mod_pad_length = input_shape[layer][1]

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		# get the pairwise projection and then extract the ITRs
		for i, ex in enumerate(csv_contents):
			print("COnverting IAD to ITR {:6d}/{:6d}".format(i, len(csv_contents)))

			f = np.load(csv_contents['iad_path_'+str(layer)])
			d, z = f["data"], f["length"]

			# prune irrelevant features
			if(pruning_keep_indexes != None):
				idx = pruning_keep_indexes[layer]
				d = d[idx]

			# clip the data for values outside of the expected range
			iad = np.clip(d, 0.0, 1.0)

			# scale the data to be between -1 and 1
			iad *= 2
			iad -= 1

			# prune file length so that it is equal to the pad_length
			if (mod_pad_length > z):
				iad = np.pad(iad, [[0,0],[0,mod_pad_length-z]], 'constant', constant_values=0)
			else:
				iad = iad[:,:mod_pad_length]

			# get the pairwise projection and then extract the ITRs
			pairwise_projections = sess.run(itr_extractor, feed_dict = {ph: data[i]})
			itr = np.array(extract_itrs(pairwise_projections[:, :z+1]))

			# save ITR
			label_path = os.path.join(ITR_DATA_PATH, ex['label_name'])
			if(not os.path.exists(label_path)):
				os.makedirs(label_path)

			file_location = os.path.join(ex['label_name'], ex['example_id'])
			itr_file = os.path.join(ITR_DATA_PATH, file_location+"_"+str(layer)+".npz")
			np.savez(filename, data=itr, label=ex['label'])

	tf.reset_default_graph()

if __name__ == '__main__':
	#provide filenames and generate and save the ITRs into a nump array
	try:
		csv_contents = [ex for ex in read_csv(FLAGS.csv_filename) if ex['dataset_id'] <= FLAGS.dataset_id]
	except:
		print("Cannot open CSV file: "+ FLAGS.csv_filename)

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict
	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):
			iad_file = os.path.join(IAD_DATA_PATH, file_location+"_"+str(layer)+".npz")
			assert os.path.exists(filename), "Cannot locate IAD file: "+ iad_file
			ex['iad_path_'+str(layer)] = iad_file

	# get the (depth, index) locations of which features to retain
	pruning_keep_indexes = None
	if(FLAGS.feature_retain_count and FLAGS.dataset_id):
		ranking_file = os.path.join(IAD_DATA_PATH, "feature_ranks_"+str(FLAGS.dataset_id * 25)+".npz")
		assert os.path.exists(filename), "Cannot locate Feature Ranking file: "+ ranking_file
		pruning_keep_indexes = get_top_n_feature_indexes(ranking_file, FLAGS.feature_retain_count)

	# Generate the ITRS, go by layer for efficiency
	for layer in range(5):
		extract_itrs_by_layer(csv_contents, layer, pruning_keep_indexes)
