# Cython-Filter-Files


Need to install Openmp, Gsl and Cython, I suggest you do it using conda. 
After that you need to go to setup.py and change the direction of the include and lib folders to your anaconda folders or where you have installed the packages.
Then install with:

python setup.py build_ext --inplace

if everything works I will modify it to install globally.
