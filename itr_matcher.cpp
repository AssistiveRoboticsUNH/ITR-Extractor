#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <unordered_set>
#include <map>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <thread> 
#include <string>
#include <fstream>
#include <sstream>

/*
int core_filters[13][2] =
	{
	{  9, -1 },//meets
	{  6, -1 },//metBy
	{  5, 11 },//starts
	{  5, 14 },//startedBy
	{  7, 10 },//finishes
	{ 13, 10 },//finishedBy
	{ 13, 11 },//overlaps
	{  7, 14 },//overlapedBy
	{  7, 11 },//during
	{ 13, 14 },//contains
	{  8,  1 },//before
	{  2,  4 },//after
	{  5, 10 },//equals
	};
*/

#define NUM_THREADS 1

namespace p = boost::python;
namespace np = boost::python::numpy;

struct thread_data{
	np::ndarray* projection;
	int num_pw_comb; 
	int proj_len;
	boost::python::list* itr_counts_src;
	int start_chunk;
	pthread_mutex_t* mut;
};

typedef int filter_type;

struct filters{
	std::unordered_set<filter_type> confounding_filters;
	std::map<filter_type, filter_type> short_filters;
	std::map<filter_type, std::map<filter_type, filter_type> > core_filters;

	filters(){
		confounding_filters = {0, 3, 12, 15};
		short_filters = {{9,0}, {6,1}};
		core_filters = {
			{8,  { {1, 10} }},
			{2,  { {4, 11} }},
			{5,  { {10, 12}, {11, 2} , {14, 3} }},
			{7,  { {10,  4}, {11, 8} , {14, 7} }},
			{13, { {10,  5}, {11, 6} , {14, 9} }}
		};
	};
};

typedef void * (*THREADFUNCPTR)(void *);


void get_matches_cpp_thread(np::ndarray* projection, const int start, const int proj_len, int* itr_counts, const int pw_comb_id, struct filters& moments, pthread_mutex_t* mut);
void *get_itrs_thread(void *thread_args);
int thread(np::ndarray projection_arr, const int num_pw_comb, const int proj_len, boost::python::list itr_counts_src);


void get_matches_cpp_thread(np::ndarray* projection_ptr, const int start, const int proj_length, int* itr_counts, const int pw_comb_id, struct filters& moments){
	
	// convert data into a numpy array
	np::ndarray projection = p::extract<np::ndarray>((*projection_ptr)[start]);
	
	int pw_index = pw_comb_id*13;
	for(int t = 0; t < proj_length;  t++){
		
		// convert the contents of array to ints.
		filter_type item = p::extract<filter_type>(projection[t]);
		
		std::map<filter_type,filter_type>::iterator loc = moments.short_filters.find( item );
		
		if(loc != moments.short_filters.end() ){
			itr_counts[pw_index + loc->second] += 1;
		}
		
		else if (t < proj_length-1){
			std::map<filter_type,std::map<filter_type,filter_type> >::iterator loc_g = moments.core_filters.find( item );
			
			if (loc_g != moments.core_filters.end() ){

				std::map<filter_type,filter_type> sub_map = loc_g->second;

				//while subsequent values are confounding filters: skip them
				int second_pos = t+1;
				filter_type item2 = p::extract<filter_type>(projection[second_pos]);

				while( second_pos < proj_length-1 && moments.confounding_filters.find( item2 ) != moments.confounding_filters.end() )  {
					second_pos++;
					item2 = p::extract<filter_type>(projection[second_pos]);
				}
				
				std::map<filter_type,filter_type>::iterator loc = sub_map.find( item2 );
				
				if ( loc != sub_map.end() ){
					itr_counts[pw_index + loc->second] += 1;
				}
			}
		}
	}	
}

void *get_itrs_thread(void *thread_args){

	// unpack thread variables
	struct thread_data *my_data = (struct thread_data *) thread_args;
	
	// figure out which ITRs are present for our data
	struct filters moments;
	
	// setup ITR list
	int* itr_counts = new int[(my_data->num_pw_comb*13)]; 
	for(int i = 0; i < my_data->num_pw_comb*13; i++){
		itr_counts[i] = 0;
	}

	// figure out which ITRs are present for our data
	for (int pw_comb = 0; pw_comb < my_data->num_pw_comb; pw_comb++){
		
		get_matches_cpp_thread(
			my_data->projection,
			my_data->start_chunk + pw_comb, 
			my_data->proj_len,
			itr_counts, 
			pw_comb,
			moments);
	}

	// update the itr_counts_src with learned values
	pthread_mutex_lock(my_data->mut);
	for (int pw_comb = 0; pw_comb < my_data->num_pw_comb; pw_comb++){
		for (int i = 0; i < 13; i++){
			(*my_data->itr_counts_src)[ ((my_data->start_chunk + pw_comb) * 13) + i] = itr_counts[pw_comb*13 + i];
		}
	}
	pthread_mutex_unlock(my_data->mut);

	// free un used memory
	delete[] itr_counts;
}


int thread(np::ndarray projection_arr, const int num_pw_comb, const int proj_len, boost::python::list itr_counts){

	// initialize mutex (might be possible to remove later)
	pthread_mutex_t mutex;
	pthread_mutex_init(&mutex, NULL);

	//vars for threads
	pthread_t threads[NUM_THREADS];
	struct thread_data td[NUM_THREADS];

	//size of data in each thread
	int chunk;
	if(num_pw_comb%NUM_THREADS == 0){
		chunk = num_pw_comb/NUM_THREADS;
	}
	else{
		std::cout << "num_pw_comb " << num_pw_comb << " doesn't divide evenly with num_threads " << NUM_THREADS << std::endl;
		return 0;
	}

	for(int i = 0; i < NUM_THREADS; i++){

		td[i].projection = &projection_arr; 
		td[i].num_pw_comb = chunk; 
		td[i].proj_len = proj_len; 
		td[i].itr_counts_src = &itr_counts;
		td[i].start_chunk = i * chunk;
		td[i].mut = &mutex;

		// run process
		pthread_create(&threads[i], NULL, (THREADFUNCPTR) &get_itrs_thread, &td[i]);
	}
	
	//make sure all threads finish
	for(int i = 0; i < NUM_THREADS; i++){
		pthread_join(threads[i], NULL);
	}

	return 1;
}

using namespace boost::python;

BOOST_PYTHON_MODULE(itr_matcher)
{
	Py_Initialize();
	np::initialize();
    def("thread", thread);
}
