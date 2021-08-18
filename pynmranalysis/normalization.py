""" Normalization methods

This script allows the user to perform NMR spesific methods 

This script requires that `pandas` and 'numpy'   be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following methods:

    * Mean normalizastion - Each spectrum is divided by its mean so that its mean becomes 1
    * Median normalization  - Each spectrum is divided by its mean so that its median becomes 1
    * Quantile normalization - Each spectrum is divided by its mean so that its first quartile becomes 1
    * Peak normalization Each spectrum divided by the maximum peak in peak range 
    * PQN normalization - Perform Probabilistic Quotient Normalization
"""

import pandas as pd
import numpy as np



def mean_normalization(spectrum, verbose=False):
    """
            Each spectrum is divided by its mean so that its mean becomes 1
            :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
            :type pandas dataframe or numpy array 
            :param verbose : if set to True return the factor defaults to false
            :type bool
            :return: noemalized spectrum
            :rtype pandas dataframe or numpy array
            """
    # Data initialisation and checks
    assert type(verbose) == bool , "verbose must be boolean"
    # Normalization
    # get the factor to devide the data
    factor = spectrum.mean(axis=1)
    # if verbos equal to true print the factor
    if verbose == True :
      print("factors : ", factor)

    #create new dataframe to store the data
    new_data = pd.DataFrame(index=spectrum.index , columns=spectrum.columns)
    for i in range (spectrum.values.shape[0]) :
      new_data.iloc[i] =spectrum.iloc[i] / factor.iloc[i]
    return new_data

# median normalization

def median_normalization(spectrum, verbose=False):
    """
                Each spectrum is divided by its mean so that its median becomes 1
                :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                :type pandas dataframe or numpy array
                :param verbose : if set to True return the factor defaults to false
                :type bol
                :return: normalizaed spectrum
                :rtype: pandas dataframe or numpy array
                """
    # Data initialisation and checks
    assert type(verbose) == bool, "verbose must be boolean"
    # Normalization
    # get the factor to devide the data
    factor = spectrum.median(axis=1)
    if verbose == True:
        print("factors : ", factor)
    # create new dataframe to store the data
    new_data = pd.DataFrame(index=spectrum.index, columns=spectrum.columns)
    for i in range(spectrum.values.shape[0]):
        new_data.iloc[i] = spectrum.iloc[i] / np.abs(factor.iloc[i])
    return new_data

#first quartile normalization

def quantile_normalization(spectrum, verbose=False):
    """
                Each spectrum is divided by its mean so that its median becomes 1
                :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                :type pandas dataframe or numpy array
                :param verbose : if set to True return the factor  defaults to false
                :type bol
                :return: normalizaed spectrum
                :rtype: pandas dataframe or numpy array
                    """
    # Data initialisation and checks
    assert type(verbose) == bool, "verbose must be boolean"
    # Normalization
    # calculate the factor that we will be used to devide the data
    factor = spectrum.quantile(0.25 , axis=1)
    if verbose == True :
      print("factors : ", factor)
    # create new dataframe to store the data
    new_data = pd.DataFrame(index=spectrum.index , columns=spectrum.columns)
    for i in range (spectrum.values.shape[0]) :
      new_data.iloc[i] =spectrum.iloc[i] / np.abs(factor.iloc[i])
    return new_data

#peak normalization

def peak_normalization(spectrum, peak_range=[3.05, 4.05], verbose=False):
    """
                        Each spectrum is divided by its mean so that its median becomes 1
                        :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                        :type pandas dataframe or numpy array
                        :param peak_range: the ppm range containing the specified peak defaults to [3.05, 4.05]
                        :type list of lenght of 2 
                        :param verbose : if set to True return the factor defaults to false
                        :type bol
                        :return: normalizaed spectrum
                        :rtype: pandas dataframe or numpy array
                        
                   
                        """
    assert type(verbose) == bool
    assert isinstance(peak_range, list) and len(peak_range) == 2 and isinstance(peak_range[0], float) and isinstance(
        peak_range[1], float), 'peak range must be list of 2 real number'

    # Normalization
    #get the ppm raneg from the columns
    ppm = spectrum.columns  # ppm values
    interval = []  # ppm range inckude un the peak range interval

    for i in ppm:
        if i > np.min(peak_range) and i < np.max(peak_range):
            interval.append(i)
    # get the data in peak zone
    data_in_zone = spectrum[interval]  # spectrum data include in the peak range
    # get the index of the peaks
    peak_in_zone = data_in_zone.idxmax(axis=1)
    #get the factor that will be used to devide the data
    factor = pd.DataFrame()
    for idx, max_idx in zip(peak_in_zone.index, peak_in_zone.values):
        factor.loc[idx, 'max'] = data_in_zone.loc[idx, max_idx]

    # print the factor
    if verbose == True:
        print("factors : ", factor)
    #creata a new datafrae to stor the normolized data
    new_data = pd.DataFrame(index=spectrum.index, columns=spectrum.columns)
    for i in range(spectrum.values.shape[0]):
        new_data.iloc[i] = spectrum.iloc[i] / factor.iloc[i].values[0]
    return new_data


def PQN_normalization(spectrum , ref_norm = "median" , verbose = False) :
  """
                        Perform Probabilistic Quotient Normalization
                        First a total area normalization should be done before PQN
                        is applied
                        Each spectrum is divided by its mean so that its median becomes 1
                        :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                        :type pandas dataframe or numpy array
                        :param ref_norm: If ref.norm is "median" or "mean", will use the median or the mean spectrum as
                         the reference spectrum ; if it is a single number, will use the spectrum located at that row
                         in the spectral matrix defaults to 'median'
                        :type str
                        :param verbose : if set to True return the factor defaults to false
                        :type bol
                        :return: normalizaed spectrum
                        :rtype: pandas dataframe or numpy array
                        
                        
                        """
  assert type(verbose) == bool , "verbose must be boolean"
  assert ref_norm in ["median" , "mean"] or ref_norm in np.arange(len(spectrum.index)) , "ref_norm is not available "
  # Normalization

  # slect the refrence specctrum
  if ref_norm == "median":
      ref_spec = spectrum.median(axis=0)
  elif ref_norm == "mean":
      ref_spec = spectrum.mean(axis=0)
  else:
      ref_spec = spectrum.iloc[ref_norm]

  # calculate the quotion
  quotion = spectrum.T.div(ref_spec.values, axis=0)
  factor = quotion.median(axis=1)
  new_data = spectrum.div(factor.values, axis=1)
  if verbose == True:
      print(factor)
  return new_data

