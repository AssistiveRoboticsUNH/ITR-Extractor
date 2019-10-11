from multiprocessing import Process

def f(model_type, dataset_dir, csv_file, pad_length, dataset_id, feature_retain_count, gpu, gpu_memory):
	cmd = "python extract_itrs.py "+model_type+" "+dataset_dir+" "+csv_file+" "+pad_length+" --dataset_id "+dataset_id+" --feature_retain_count "+feature_retain_count+" --gpu "+gpu+" --gpu_memory "+gpu_memory
	print(cmd)
    #os.system()

if __name__ == '__main__':

	model_type = "i3d"
	dataset_dir = "~/HMDB-51"
	csv_file = "~/HMDB-51/hmdb.csv"
	pad_length = str(256)
	feature_retain_count = str(128)
	gpu = str(0)
	gpu_memory = str(0.25)

	procs = []
	for dataset_id in range(4, 0, -1):
    	p = Process(target=f, args=(model_type, dataset_dir, csv_file, pad_length, dataset_id, feature_retain_count, gpu, gpu_memory, ))
    	p.start()
    
    for p in procs:
   		p.join()