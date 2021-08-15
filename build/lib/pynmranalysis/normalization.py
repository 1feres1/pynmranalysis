import pandas as pd
import numpy as np

# mean normalizastion

def mean_normalization(spectrum, verbose=True):
    """
            , returns normalized spectrum where each spectrum is devided by its mean
            :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
            :param verbose : if set to True return the factor
            :return: noemalized spectrum
            """
    # Data initialisation and checks
    assert type(verbose) == bool , "verbose must be boolean"
    # Normalization

    factor = spectrum.mean(axis=1)
    if verbose == True :
      print("factors : ", factor)
    new_data = pd.DataFrame(index=spectrum.index , columns=spectrum.columns)
    for i in range (spectrum.values.shape[0]) :
      new_data.iloc[i] =spectrum.iloc[i] / factor.iloc[i]
    return new_data

# median normalization

def median_normalization(spectrum, verbose=True):
    """
                , returns normalized spectrum where each spectrum is devided by its median
                :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                :param verbose : if set to True return the factor
                :return: normalizaed spectrum
                """
    # Data initialisation and checks
    assert type(verbose) == bool, "verbose must be boolean"
    # Normalization

    factor = spectrum.median(axis=1)
    if verbose == True:
        print("factors : ", factor)
    new_data = pd.DataFrame(index=spectrum.index, columns=spectrum.columns)
    for i in range(spectrum.values.shape[0]):
        new_data.iloc[i] = spectrum.iloc[i] / np.abs(factor.iloc[i])
    return new_data

#first quartile normalization

def quantile_normalization(spectrum, verbose=True):
    """
                    , returns normalized spectrum where each spectrum is devided by its first quartile
                    :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                    :param verbose : if set to True return the factor
                    :return: normalizaed spectrum
                    """
    # Data initialisation and checks
    assert type(verbose) == bool, "verbose must be boolean"
    # Normalization

    factor = spectrum.quantile(0.25 , axis=1)
    if verbose == True :
      print("factors : ", factor)
    new_data = pd.DataFrame(index=spectrum.index , columns=spectrum.columns)
    for i in range (spectrum.values.shape[0]) :
      new_data.iloc[i] =spectrum.iloc[i] / np.abs(factor.iloc[i])
    return new_data

#peak normalization

def peak_normalization(spectrum, peak_range=[3.05, 4.05], verbose=True):
    """
                        , returns normalized spectrum where each spectrum is devided by maximum peak in each spectrum
                        :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                        :param peak_range: the ppm range containing the specified peak
                        :param verbose : if set to True return the factor
                        :return: normalizaed spectrum
                        """
    assert type(verbose) == bool
    assert isinstance(peak_range, list) and len(peak_range) == 2 and isinstance(peak_range[0], float) and isinstance(
        peak_range[1], float), 'peak range must be list of 2 real number'

    # Normalization

    ppm = spectrum.columns  # ppm values
    interval = []  # ppm range inckude un the peak range interval

    for i in ppm:
        if i > np.min(peak_range) and i < np.max(peak_range):
            interval.append(i)

    data_in_zone = spectrum[interval]  # spectrum data include in the peak range

    peak_in_zone = data_in_zone.idxmax(axis=1)
    factor = pd.DataFrame()
    for idx, max_idx in zip(peak_in_zone.index, peak_in_zone.values):
        factor.loc[idx, 'max'] = data_in_zone.loc[idx, max_idx]
    if verbose == True:
        print("factors : ", factor)
    new_data = pd.DataFrame(index=spectrum.index, columns=spectrum.columns)
    for i in range(spectrum.values.shape[0]):
        new_data.iloc[i] = spectrum.iloc[i] / factor.iloc[i].values[0]
    return new_data

def PQN_normalization(spectrum , ref_norm = "median" , verbose = False) :
  """
                        , returns normalized spectrum where each spectrum is devided by
                        :param spectrum: dataframe of the spectrun values index are the cases and columns are the ppm value
                        :param ref_norm: If ref.norm is "median" or "mean", will use the median or the mean spectrum as
                         the reference spectrum ; if it is a single number, will use the spectrum located at that row
                         in the spectral matrix;
                        :param verbose : if set to True return the factor
                        :return: normalizaed spectrum
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
