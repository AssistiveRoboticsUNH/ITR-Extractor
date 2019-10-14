from multiprocessing import Process
from itr_evaluator import main






def f(dataset_dir, csv_filename, num_classes, dataset_id, batch_size, epochs, alpha, gpu):
	main(dataset_dir, csv_filename, num_classes, dataset_id, batch_size, epochs, alpha, gpu)

if __name__ == '__main__':


	import argparse
	parser = argparse.ArgumentParser(description="Ensemble model processor")
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	#parser.add_argument('dataset_id', nargs='?', type=int, help='the dataset_id used to train the network. Is used in determing feature rank file')

	#parser.add_argument('--train', default='', help='.list file containing the train files')
	#parser.add_argument('--test', default='', help='.list file containing the test files')
	parser.add_argument('--batch_size', type=int, default=10, help='.list file containing the test files')
	parser.add_argument('--epochs', type=int, default=10, help='.list file containing the test files')
	parser.add_argument('--alpha', nargs='?', type=int, default=1e-4, help='the maximum length video to convert into an IAD')

	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	

	procs = []
	for dataset_id in range(4, 0, -1):
		p = Process(target=f, args=(FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes, 
		dataset_id, 
		FLAGS.batch_size, 
		FLAGS.epochs, 
		FLAGS.alpha, 
		FLAGS.gpu, ))
		p.start()
		p.join()
	
		