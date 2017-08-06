# Latural Language Processing (NLP), Latent Dirichlet Allocation (LDA) & Non-negative Matrix Factorization (NMF)

Python library for analyze corporates' shareholders letter. This uses two natural language processing methods (LDA & NMF).

For more information about LDA and NMF, click the links below.

-  LDA: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

-  NMF: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization


# Prerequisites
This NLP applications assumes you have installed Python 3.x for Windows and corresponding Pips. It may work, but is untested on Python 2.x and other operating systems.

# Installation
This application uses NumPy, scikit-learn and matplotlib libraries.
-  Numpy (>= 1.6.1) is a python library for support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
-  scikit-learn (>= 2.6 or >= 3.3) is a python library for machine learning. 
-  Matplotlib is a Python 2D plotting library.

For Windows users download Numpy and sklearn whl files correspond your Windows and Python versions from the link. 
-  http://www.lfd.uci.edu/~gohlke/pythonlibs/

Mac users can download the files and refer how to install them from 
-  scikit-learn: https://pypi.python.org/pypi/scikit-learn/0.18.1
-  NumPy: https://pypi.python.org/pypi/numpy

Download the files in the C:\Python36\Lib\site-packages repository.
```shell
pip install C:\python36\Lib\site-packages\File_Name (For example, C:\python36\Lib\site-packages\numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl)
 or
pip install numpy
pip install sklearn
```
Install matplotlib
```shell
pip install matplotlib
```

download the [example `products.csv` file](https://raw.githubusercontent.com/prof-rossetti/nyu-info-2335-70-201706/master/projects/crud-app/products.csv) and save it as `data/products.csv`.

