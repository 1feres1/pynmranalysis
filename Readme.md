
# Pynmranalysis
## Python library for NMR preprocessing and analysis


<p align="center">
    <img src="https://github.com/1feres1/pynmranalysis/blob/main/PyNMRanalysis-logos.jpeg" width="300" height="100">
</p>


[![Build Status](https://travis-ci.com/1feres1/pynmranalysis.svg?branch=main)](https://travis-ci.com/1feres1/pynmranalysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A5qS1ObiiYBXmPnlecCTxzV41BzQ3fG6?usp=sharing)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/1feres1/pynmranalysis/releases/)

**Pynmranalysis** make it possible to work with **1H NMR spectrum**  using **python**.
It makes analysing the spectrum more effective by offering many **preprocessing functions** and  makes it easy to perform **statistical modeling** with great plots.

With **Pynmranalysis**, you are no longer restricted to use R or matlab to work with **NMR signals** !



## Installation


Install the pachage with pip command:


```sh
pip install pynmranalysis
```
You may also install directly from this repository for the current master:
```sh
pip install git+git://github.com/1feres1/pynmranalysis.git
```
##### Required dependencies:
- numpy == 1.20.3 
- pandas == 1.2.4 
- scipy == 1.6.3
- scikit-learn == 0.24.2
- matplotlib == 3.4.2

## Online Demo
The following notebook shows you how to use the main functions of our library.
This includes performing the **preprocessing steps** on 1H NMR dataset, scaling this data using NMR specific **normalization function** and finaly performing statistical analysis methodes like **PCA** and **PLS-DA**.

You can test it yourself via this **link**:

https://colab.research.google.com/drive/1A5qS1ObiiYBXmPnlecCTxzV41BzQ3fG6?usp=sharing

## How to use 

#### Preprocessing
Preprocessing is a set of operations applyed to raw data in order to prepare it for further analysis

We will use a CSV file containing 1H-NMR spectra for 71 serum samples of patients with coronary heart disease (CHD) and healthy controls,located in example/CHD.csv in the exemple folder of this repository

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

#### Binning / Bucketing
In order to reduce the data dimensionality binning is commonly used. In binning, the spectra are divided into bins (so called buckets) and the total area within each bin is calculated to represent the original spectrum. Here is an example:


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

#### Region Removal 
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
**Note** : The implementation of these functions is similar to R's  PepsNMR library [[1]](#1).
#### Normalization

The comparison between the spectra is impossible without prior normalization. Therefore, a normalization step allows the data from all the spectra to be directly comparable

##### Mean Normalization 
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
##### Median Normalization
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
##### Quantile Normalization
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
##### Peak Normalization
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
##### PQN Normalization
We used the definition from Dieterle et al [[2]](#2). If ref_norm is "median" or "mean", we will use the median or the mean spectrum as the reference spectrum ; if it is a single number, will use the spectrum located at that row in the spectral matrix ,it defines manually the reference spectrum.

```python
from pynmranalysis.normalization import PQN_normalization
norm_spectrum = PQN_normalization(r_spectrum ,ref_norm = "median" , verbose=False)
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
**Note** : The implementation of these functions is similar to  R's PepsNMR library [[1]](#1).
#### statistical analysis

##### PyPCA 

Principal component analysis, or **PCA**, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.

A pickle file containing 1H-NMR spectra for 64 serum samples of patients with two groups of digestive diseases, biliary/Pancreatic Disease and Intestinal Diseases is located in digestive_disease_data.pkl in the exemple folder of this repository. 

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
PyPCA class and it's methods are used to perform PCA.
``` python 
from pynmranalysis.analysis import PyPCA
#create pypca instance 
pca = PyPCA(n_comps=3) 
#fit the model to data
pca.fit(spectrum)
```
The score plot is the projection of samples in the dataset in lower dimension space of the first 2 components of the model 

``` python 
pca.score_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/score_plot.PNG" >
The scree plot is a graph that shows each component of the PCA model with their explained variance.

``` python 
pca.scree_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/scree_plot.PNG" >
Outiler plot is a plot that calculates the index of the outliers in the data and plot them with a different color.

``` python 
pca.outlier_plot()
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/outlier_plot.PNG" >
The target plot is a scatter plot that shows the projection of each simple in the first 2 components with colors that matchs their classses in the target variable.

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
The interia plot is a paired barbot that shows R2Y (goodness of the fit ) score and R2Y (goodnes of predection with cross validation)

``` python 
plsda.inertia_barplot(spectrum, target)
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/interia%20plot.PNG" >
PLSDA score plot is a scatter plot that shows the projection of simples in the first 2 latent variables.

``` python 
plsda.score_plot(target)
```
<img src="https://github.com/1feres1/pynmranalysis/blob/main/exemple/plsda_score_plot.PNG" >

Note : The implementation of these functions is similar to R's PepsNMR library [[3]](#3).

### License

MIT



## Reference
<a id="1">[1]</a> 
PepsNMR for 1 H NMR metabolomic data pre-processing Manon Martin , Benoît Legat

<a id="2">[2]</a>
Probabilistic Quotient Normalization as Robust Method to Account for Dilution of Complex Biological Mixtures. Application in 1H NMR Metabonomics

<a id="3">[3]</a> 
Partial least square for discrimination Matthew Barker1 and William Rayens



