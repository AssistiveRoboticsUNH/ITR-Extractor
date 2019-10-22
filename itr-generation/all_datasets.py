from multiprocessing import Process
from extract_itrs import main




def f(model_type, dataset_dir, csv_file, pad_length, dataset_id, feature_retain_count, gpu, gpu_memory):
	main(model_type, dataset_dir, csv_file, pad_length, dataset_id, feature_retain_count, gpu, gpu_memory)

if __name__ == '__main__':


	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use: I3D')

	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')

	#feature pruning command line args
	#parser.add_argument('--dataset_id', nargs='?', type=int, default=4, help='the dataset_id used by the csv file')
	parser.add_argument('--feature_retain_count', nargs='?', type=int, default=-1, help='the number of features to remove')

	parser.add_argument('--gpu_memory', nargs='?', type=float, default=0.5, help='how much of the GPU should the process consume')
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	procs = []
	for dataset_id in range(4, 0, -1):
		p = Process(target=f, args=(FLAGS.model_type, FLAGS.dataset_dir, FLAGS.csv_filename, FLAGS.pad_length, dataset_id, FLAGS.feature_retain_count, FLAGS.gpu, FLAGS.gpu_memory, ))
		p.start()
	
	for p in procs:
		p.join()