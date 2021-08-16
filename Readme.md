
# Pynmranalysis
## python library for NMR preprocessing and analysis

<img src="https://github.com/1feres1/pynmranalysis/blob/main/PyNMRanalysis-logos.jpeg" width="300" height="100">



[![Build Status](https://travis-ci.com/1feres1/pynmranalysis.svg?branch=main)](https://travis-ci.com/1feres1/pynmranalysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A5qS1ObiiYBXmPnlecCTxzV41BzQ3fG6?usp=sharing)
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.org/project/pynmranalysis/)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/1feres1/pynmranalysis/releases/)

pynmranalysis has the ability to work with 1H NMR spectrum and offers many preprocessing functions that makes analysing the spectrum more effective
also it can be used to perform statistical modeling with great plots
- preprocessing steps
- Normalization 
- Statistical analysis


## Installation


Install the pachage with pip command


```sh
pip install pynmranalysis
```
You may also install directly from this repository for the current master:
```sh
pip install git+git://github.com/1feres1/pynmranalysis.git
```
## How to use 
the following python script shows you how to use the main functions of our library
demo link:
https://colab.research.google.com/drive/1A5qS1ObiiYBXmPnlecCTxzV41BzQ3fG6?usp=sharing
### preprocessing
A CSV file containing 1H-NMR spectra for 71 serum samples of patients with coronary heart disease (CHD) and healthy controls is located in CHD.csv in the exemple folder of this repository

```python
# import 
import matplotlib.pyplot as plt
import pandas as pd
#read coronary heart disease data
spectrum = pd.read_csv("CHD.csv")
#convert columns from string to real numbers
columns = [float(x) for x in spectrum.columns]
spectrum.columns  = columns
```

##### binning / bucketing
In order to reduce the data dimensionality binning is commonly used. In binning the spectra are divided into bins (so called buckets) and the total area within each bin is calculated to represent the original spectrum


```python
from pynmranalysis.nmrfunctions import binning
binned_data = binning(spectrum ,width=True ,  bin_size = 0.04 , int_meth='simps' , verbose=False)
```
```python

fig , axs = plt.subplots(2,1 , figsize = (16,5))
fig.tight_layout()
axs[0].plot(spectrum.iloc[0] )
axs[0].set(title = 'spectrum before binning')
axs[1].plot(binned_data.iloc[0] )
axs[1].set(title = 'spectrum after binning')
plt.show()
```

##### region removal 
By default, this step sets to zero spectral areas that are of no interest or have a sigificant and unwanted amount of variation (e.g. the water area).


```python
from pynmranalysis.nmrfunctions import region_removal
r_spectrum = region_removal(spectrum=binned_data )
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(binned_data.iloc[0] )
axs[0].set(title = 'spectrum before region removal')
axs[1].plot(r_spectrum.iloc[0] )
axs[1].set(title = 'spectrum after region removal')
plt.show()

```
### normalization
##### mean normalization 
Each spectrum is divided by its mean so that its mean becomes 1.


```python
from pynmranalysis.normalization import median_normalization
norm_spectrum = median_normalization(r_spectrum , verbose=False)
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(r_spectrum.iloc[0] )
axs[0].set(title = 'spectrum before normalization')
axs[1].plot(norm_spectrum.iloc[0] )
axs[1].set(title = 'spectrum without normalization')
plt.show()

```
##### median normalization
Each spectrum is divided by its median so that its median becomes 1.

```python
from pynmranalysis.normalization import quantile_normalization
norm_spectrum = quantile_normalization(r_spectrum , verbose=False)
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(r_spectrum.iloc[0] )
axs[0].set(title = 'spectrum before normalization')
axs[1].plot(norm_spectrum.iloc[0] )
axs[1].set(title = 'spectrum without normalization')
plt.show()
```
##### quantile normalization
Each spectrum is divided by its first quartile so that its first quartile becomes 1.

```python
from pynmranalysis.normalization import mean_normalization
norm_spectrum = mean_normalization(r_spectrum , verbose=False)
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(r_spectrum.iloc[0] )
axs[0].set(title = 'spectrum before normalization')
axs[1].plot(norm_spectrum.iloc[0] )
axs[1].set(title = 'spectrum without normalization')
plt.show()
```
##### peak normalization
Each spectrum is divided by the value of the peak of the spectrum contained between "peak_range" inclusive (i.e. the maximum value of spectral intensities in that interval).

```python
from pynmranalysis.normalization import peak_normalization
norm_spectrum = peak_normalization(r_spectrum , verbose=False)
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(r_spectrum.iloc[0] )
axs[0].set(title = 'spectrum before normalization')
axs[1].plot(norm_spectrum.iloc[0] )
axs[1].set(title = 'spectrum without normalization')
plt.show()
```
##### PQN normalization
Probabilistic Quotient Normalization from Dieterle et al. (2006). If ref.norm is "median" or "mean", will use the median or the mean spectrum as the reference spectrum ; if it is a single number, will use the spectrum located at that row in the spectral matrix; if ref.norm is a numeric vertor of length equal to the number of spectral variables, it defines manually the reference spectrum.

```python
from pynmranalysis.normalization import PQN_normalization
norm_spectrum = PQN_normalization(r_spectrum , verbose=False)
```
```python
fig , axs = plt.subplots(2,1, figsize = (16,5))
fig.tight_layout()
axs[0].plot(r_spectrum.iloc[0] )
axs[0].set(title = 'spectrum before normalization')
axs[1].plot(norm_spectrum.iloc[0] )
axs[1].set(title = 'spectrum without normalization')
plt.show()
```

### statistical analysis
#### PCA 
A pickle file containing 1H-NMR spectra for 64 serum samples of patients with two groups of disgstive diseases bliary/Pancreatic Disease and Intestinal Diseases is located in digestive_disease_data.pkl in the exemple folder of this repository 

```python
# import 
import matplotlib.pyplot as plt
import pandas as pd
#read data
data = pd.read_pickle('digestive_disease_data.pkl')
# split data into predictive variables (spectrums) and target varibles (digestive disease group)
# target -->  1 :Biliary/Pancreatic Diseases | 0 : Intestinal Diseases
spectrum = data.iloc[ : , :-1]
target = data.iloc[ : , -1].values
```

#### PyPCA 

Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed

``` python 
from pynmranalysis.analysis import PyPCA
#create pypca instance 
pca = PyPCA(n_comps=3) 
#fit the model to data
pca.fit(spectrum)
```
score plot is the projection of samples in the data set in lower dimention spce of the first 2 componants of the 

``` python 
pca.score_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/score_plot.PNG" >
scree plot is agraph that show each componant of the pca model with their explained variance

``` python 
pca.scree_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/scree_plot.PNG" >
outiler plot is a plot that calculate index of outliers in the data and plot them with different color

``` python 
pca.outlier_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/outlier_plot.PNG" >
target plot is a scatter plot that shows the projection of each simple in the first 2 componants with 
colors that much their classses in the target variable

``` python 
pca.target_plot(target)
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/target_plot.PNG" >

#### PyPLS_DA

Partial least squares-discriminant analysis (PLS-DA) is a versatile algorithm that can be used for predictive and descriptive modelling as well as for discriminative variable selection.

``` python 
from pynmranalysis.analysis import PyPLS_DA
#create pyplsda instance 
plsda = PyPLS_DA(ncomps=3) 
#fit the model to data
plsda.fit(spectrum , target)
```
interia plot is a paired barbot that shows R2Y (goodness of the fit ) score and R2Y (goodnes of predection with cross validation)

``` python 
plsda.inertia_barplot(spectrum, target)
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/interia%20plot.PNG" >
PLSDA score plot is a scatter plot that shows the projection of simples in the first 2 latent variables

``` python 
plsda.score_plot(target)
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/plsda_score_plot.PNG" >

### License

MIT


