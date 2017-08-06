# Latural Language Processing (NLP), Latent Dirichlet Allocation (LDA) & Non-negative Matrix Factorization (NMF)

This application analyzes five companies' (Adobe, ea, Aspen, Compuware and Citrix) shareholders letter from 1993 to 2003 with two different natural language processing methods (LDA & NMF).

With two methods, it shows ten topic models in 48 letters, topic shares associated with each company. In addition, it visualizes the results with a heatmap and a topic distance map. 

For more information about LDA and NMF, click the links below.

-  LDA: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

-  NMF: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization


# Prerequisites
This NLP applications assumes you have installed Python 3.x for Windows and corresponding Pips. It may work, but is untested on Python 2.x and other operating systems.

# Installation
This application uses NumPy, scikit-learn and matplotlib libraries.
-  Numpy (>= 1.6.1) is a python library for support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
-  scikit-learn (>= 2.6 or >= 3.3) is a python library for machine learning. 
-  Matplotlib (>= 2.7 or >= 3.4) is a Python 2D plotting library.

For Windows users, download Numpy and sklearn whl files correspond your Windows and Python versions from the link. 
-  http://www.lfd.uci.edu/~gohlke/pythonlibs/

Download the files in the C:\Python36\Lib\site-packages repository.


Mac users can download the files and refer how to install them from 
-  scikit-learn: https://pypi.python.org/pypi/scikit-learn/0.18.1
-  NumPy: https://pypi.python.org/pypi/numpy

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

# Usage
Run the application:
```shell
-  LDA model: python app\NLP_LDA.py
-  NMF model: python app\NLP_NMF.py
```

Run tests:
```shell
LDA script: pytest test\test_NLP_LDA.py
NMF script: pytest test\test_NLP_NMF.py
or
pytest  # test two scripts at the same time
```


