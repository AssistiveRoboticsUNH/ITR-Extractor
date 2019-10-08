import argparse
import numpy as np
import tensorflow as tf

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Ensemble model processor")
parser.add_argument('model', help='model to save (when training) or to load (when testing)')
parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
parser.add_argument('prefix', help='"train" or "test"')
parser.add_argument('itr_prefix', help='prefix used when generating the ITRs')

#parser.add_argument('--train', default='', help='.list file containing the train files')
#parser.add_argument('--test', default='', help='.list file containing the test files')
parser.add_argument('--batch_size', type=int, default=10, help='.list file containing the test files')
parser.add_argument('--epochs', type=int, default=10, help='.list file containing the test files')

parser.add_argument('--gpu', default="0", help='gpu to run on')
parser.add_argument('--v', default=False, help='verbose')

args = parser.parse_args()

def file_io(file):
	f = np.load(file)
	data, label = f["data"], f["label"]
	return data, label 




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

def train(model_name, num_classes, input_shape, train_data, train_label, test_data, test_label):
	placeholders, ops = model(num_classes, input_shape)
	saver = tf.train.Saver()

	val_accs, tst_accs = [], []
	val_losses, tst_losses = [], []

	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		num_iter = train_label.shape[0] * args.epochs

		for i in range(num_iter):

			# train op
			idx = np.random.randint(0, len(train_label), size=args.batch_size)
			sess.run(ops["train"], feed_dict={placeholders["input"]: train_data[idx], placeholders["output"]: train_label[idx]})

			if(i % 100 == 0):
				print("Iteration: {:6d}/{:6d}".format(i, num_iter))
				# reset the metrics
				stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
				sess.run(tf.variables_initializer(stream_vars_valid))

				#validation accuracy
				val_acc, val_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: train_data[idx], placeholders["output"]: train_label[idx]})
				print("Validation - "),
				print("accuracy: {:.6f}".format(val_acc)),
				print(", loss: {0}".format(val_loss))

				val_accs.append(val_acc)
				tst_accs.append(val_loss)

				#test accuracy
				idx = np.random.randint(0, len(test_label), size=args.batch_size)
				tst_acc, tst_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: test_data[idx], placeholders["output"]: test_label[idx]})
				print("Test - "),
				print("accuracy: {:.6f}".format(tst_acc)),
				print(", loss: {0}".format(tst_loss))

				val_losses.append(tst_acc)
				tst_losses.append(tst_loss)



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



def test(model_name, num_classes, input_shape, test_data, test_label):
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

		for idx in range(len(test_label)):
			tst_acc, tst_loss = sess.run([ops["cumulative_accuracy"], ops["loss"]], feed_dict={placeholders["input"]: test_data[None, idx], placeholders["output"]: test_label[None, idx]})

	print("Test - "),
	print("accuracy: {:.6f}".format(tst_acc)),
	print(", loss: {0}".format(tst_loss))


if __name__ == '__main__':

	for layer in range(5):

		model_dir = os.path.join(args.model, str(layer+1).zfill(2))
		if(not os.path.exists(model_dir):
			os.makedirs(model_dir)

		train_filename = args.itr_prefix + "_train_" + str(layer) + ".npz"
		test_filename = args.itr_prefix + "_test_" + str(layer) + ".npz"

		test_data, test_label = file_io(test_filename)
		train_input_shape = [args.batch_size, test_data.shape[1], test_data.shape[2]]
		test_input_shape =  [              1, test_data.shape[1], test_data.shape[2]]

		if(args.prefix == "train"):
			train_data, train_label = file_io(train_filename)
			train(model_dir, args.num_classes, train_input_shape, train_data, train_label, test_data, test_label)
		else:
			test(model_dir, args.num_classes, test_input_shape, test_data, test_label)
