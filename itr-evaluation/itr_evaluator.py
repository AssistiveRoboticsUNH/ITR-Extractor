import numpy as np
import tensorflow as tf

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sys
sys.path.append("../iad-generation/")
from csv_utils import read_csv

"""
def file_io(file):
	f = np.load(file)
	data, label = f["data"], f["label"]
	return data, label 
"""



def model(num_classes, input_shape):

	placeholders = {
		"input": tf.placeholder(tf.float32, shape=input_shape, name="input_ph"),
		"output": tf.placeholder(tf.int32, shape=[input_shape[0]], name="output_ph")
		}

	top = placeholders["input"]
	top = tf.layers.flatten(top)
	top = tf.layers.dense(top, 1024, name="dense1")
	out = tf.layers.dense(top, num_classes, name="out")

	#---------------

	print("---> out ", out.get_shape(), placeholders["output"].get_shape())

	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=placeholders["output"],logits=out)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

	#---------------

	softmax_op = tf.nn.softmax(out)
	eval_op = tf.argmax(out, axis=1)

	with tf.name_scope('metrics'):
		acc, acc_op = tf.metrics.accuracy(labels=placeholders["output"],predictions=tf.argmax(out, 1))

	ops = {
		"train": train_op,
		"confidence": softmax_op,
		"eval": eval_op,
		"loss": tf.reduce_mean(loss),
		"accuracy": acc,
		"cumulative_accuracy": acc_op
	}

	return placeholders, ops

def get_data(ex, layer):
	f = np.load(ex['itr_path_'+str(layer)])
	return np.expand_dims(f['data'], axis=0), ex['label']

def get_batch_data(dataset, batch_size, layer):

	idx = np.random.randint(0, len(dataset), size=batch_size)

	data, label = [],[]
	for i in idx:
		data, label = get_data(dataset[i], layer)

	return np.array(data), np.array(label)

def train(model_name, num_classes, input_shape, train_data, test_data):
	placeholders, ops = model(num_classes, input_shape)
	saver = tf.train.Saver()

	#val_accs, tst_accs = [], []
	#val_losses, tst_losses = [], []

	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		num_iter = train_label.shape[0] * args.epochs

		for i in range(num_iter):

			# train op
			#idx = np.random.randint(0, len(train_label), size=args.batch_size)
			data, label = get_batch_data(train_data, batch_size, layer)
			sess.run(ops["train"], feed_dict={placeholders["input"]: data, placeholders["output"]: label})

			if(i % 100 == 0):
				print("Iteration: {:6d}/{:6d}".format(i, num_iter))
				# reset the metrics
				stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
				sess.run(tf.variables_initializer(stream_vars_valid))

				#validation accuracy
				val_acc, val_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: data, placeholders["output"]: label})
				print("Validation - "),
				print("accuracy: {:.6f}".format(val_acc)),
				print(", loss: {0}".format(val_loss))

				val_accs.append(val_acc)
				tst_accs.append(val_loss)

				#test accuracy
				data, label = get_batch_data(test_data, batch_size, layer)
				tst_acc, tst_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: data, placeholders["output"]: label})
				print("Test - "),
				print("accuracy: {:.6f}".format(tst_acc)),
				print(", loss: {0}".format(tst_loss))

				val_losses.append(tst_acc)
				tst_losses.append(tst_loss)

	'''

		# plot learning
		plt.subplot(211)
		plt.plot(np.array(val_accs))
		plt.plot(np.array(tst_accs))

		plt.suptitle('Accuracies')
		plt.subplot(212)
		plt.plot(np.array(val_losses))
		plt.plot(np.array(tst_losses))

		plt.suptitle('Losses')

		plt.legend()
		#plt.show()
		plt.savefig("accs_losses.png")


		# save model
		print("Finished training")
		saver.save(sess, model_name+"/model")
		print("Model saved")
		
	tf.reset_default_graph()
	'''


def test(model_name, num_classes, input_shape, test_data):
	placeholders, ops = model(num_classes, input_shape)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# load saved model
		ckpt = tf.train.get_checkpoint_state(model_name)
		if ckpt and ckpt.model_checkpoint_path:
			print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("load complete!")

		sess.run(tf.local_variables_initializer())

		for ex in test_label:
			data, label = get_data(ex)
			tst_acc, tst_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: data, placeholders["output"]: label})
	tf.reset_default_graph()
	print("Test - "),
	print("accuracy: {:.6f}".format(tst_acc)),
	print(", loss: {0}".format(tst_loss))

def main(dataset_dir, csv_filename, num_classes, dataset_id, batch_size, epochs, alpha, gpu):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	#provide filenames and generate and save the ITRs into a nump array
	try:
		csv_contents = [ex for ex in read_csv(csv_filename) if ex['dataset_id'] <= dataset_id]
	except:
		print("Cannot open CSV file: "+ csv_filename)

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >  0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0] 

	train_data = train_data[:5]
	test_data = test_data[:5]

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict

	dataset_id_path = os.path.join('itr', "dataset_"+str(25*dataset_id))
	itr_data_path = os.path.join(dataset_dir, dataset_id_path)

	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):

			itr_file = os.path.join(itr_data_path, file_location+"_"+str(layer)+".npz")
			assert os.path.exists(itr_file), "Cannot locate IAD file: "+ itr_file
			ex['itr_path_'+str(layer)] = itr_file



	for layer in range(5):

		'''
		model_dir = os.path.join(args.model, str(layer+1).zfill(2))
		if(not os.path.exists(model_dir)):
			os.makedirs(model_dir)
		'''

		train_input_shape = [args.batch_size, test_data.shape[1], test_data.shape[2]]
		train(model_dir, args.num_classes, train_input_shape, train_data, test_data)
		
		test_input_shape =  [              1, test_data.shape[1], test_data.shape[2]]
		test(model_dir, args.num_classes, test_input_shape, test_data)



if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description="Ensemble model processor")
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	parser.add_argument('dataset_id', nargs='?', type=int, help='the dataset_id used to train the network. Is used in determing feature rank file')

	#parser.add_argument('--train', default='', help='.list file containing the train files')
	#parser.add_argument('--test', default='', help='.list file containing the test files')
	parser.add_argument('--batch_size', type=int, default=10, help='.list file containing the test files')
	parser.add_argument('--epochs', type=int, default=10, help='.list file containing the test files')
	parser.add_argument('--alpha', nargs='?', type=int, default=1e-4, help='the maximum length video to convert into an IAD')

	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes, 
		FLAGS.dataset_id, 
		FLAGS.batch_size, 
		FLAGS.epochs, 
		FLAGS.alpha, 
		FLAGS.gpu)

	