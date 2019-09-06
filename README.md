# ITR-Extractor
Extract ITRs from IADs





To get Boost numpy working use these commands: (https://stackoverflow.com/questions/51037886/trouble-with-linking-boostpythonnumpy)

sudo apt -y install libpython2.7-dev libboost-python-dev
git clone https://github.com/ndarray/Boost.NumPy
cd Boost.Numpy
mkdir build
cd build
cmake ..
make 
sudo make install
replace in your code boost/python/numpy.hpp with boost/numpy.hpp also replace namespace np = boost::python::numpy with namespace np = boost::numpy; |
