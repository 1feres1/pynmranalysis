""" Preprocessing methods 

This script allows the user to perform preprocessing methods on NMR spectrum 

This script requires that `pandas` , 'scipy' and 'numpy'   be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following methods:

    * Binning - divide the spectrum into equal lenghts bins and integrate the peak in each bin 
    * Region Removal   - remove undesired zones from the spectrum 
 
"""

from scipy import integrate
import pandas as pd
import numpy as np


def binning(spectrum, width=False, mb=500, bin_size=0.04, int_meth="simps", verbose=False):
    """
        returns binned spectrum with every spectrum bins is the integral of the original spectrum along the
        ppm range

        :param: spectrum: dataframe of the spectrun values index are the cases and columns are the ppm values
        :type pandas dataframe or numpy array 
        :param width: if width is True we use the bin size as a ppm window for binning defaults to false
        :type bool
        :param bin_size: bin size is the ppm range of each bin defaults to 500
        :type float
        :param int_meth: if set to simps use the composite Simpson’s rule for integration and trap use the composite trapezoidal rule defaults to 'simps'
        :type str
        :param verbose: display inforamation about binning defaults to false
        :type bool

        :return: binned spectrum with columns the bins range
        :rtype: pandas dataframe or numpy array 

        """
    # Data initialisation and checks ----------------------------------------------

    assert isinstance(width, bool), "the width arg must be True or False "
    assert mb > 0 and isinstance(mb, int), " mb must be positive integer "
    assert int_meth in ["simps", "traps"], "integration methode is indifined"
    assert isinstance(verbose, bool), "verbose must be boolean"

    # Bucketting ----------------------------------------------
    # get the ppm range from the columns
    ppm = spectrum.columns
    if isinstance(ppm[0], str):
        ppm = [float(x) for x in ppm]
    # calculate the number of bins based on the bn width
    if width == True:
        nb_bins = int((max(ppm) - min(ppm)) / bin_size)
    else:
        # use he passed number of bins
        nb_bins = mb
    #create a new dataframe to store the normalized data
    new_df = pd.DataFrame()
    # devide the ppm range into buckets based on the number of bins
    buckets = np.array_split(spectrum.columns, nb_bins)
    for i, bucket in enumerate(buckets):
        # select the integration method
        if int_meth == "simps":
            x = integrate.simps(spectrum.loc[:, bucket], bucket[::-1])
        else:
            x = integrate.trapz(spectrum.loc[:, bucket], bucket[::-1])

        new_df[np.round(bucket[0], 3)] = x
    #restor ethe old indexs
    new_df.index = spectrum.index
    if verbose == True:
        print('number of bins : ', nb_bins)
    return new_df


def region_removal(spectrum, type_of_spec="manual", type_remv="Zero", intervals=[4.5, 5.1], verbose=False):
    """
          Removes the non-informative regions by setting the values of the spectra in these intervals to zero.

          :param: spectrum: dataframe of the spectrun values index are the cases and columns are the ppm values
          :type  pandas dataframe or numpy array
          :param type_of_spec: if not "manual", will automatically remove unwanted regions depending on the nature of spectra defaults to 'manual'
          :type str
          :param type_remv:  If equal to "zero", intensities are set to 0; if type.rr = "NaN", intensities are set to NaN. defaults to 'Zero'
          :type str
          :param verbose: If "True", will print processing information. defaults to bool
          :type bool
          :params intervals: List containing the extremities of the intervals to be removed. defaults to [4.5, 5.1]
          : type list of length of 2 
          :return: spectrum without the unwanted regions
          :rtype: pandas dataframe or numpy array

    """
    # Data initialisation and checks ----------------------------------------------

    assert isinstance(verbose, bool), "verbose must be boolean"
    assert type_of_spec in ['manual', "serum", "urine"], "this type of spectrum is not available "
    if isinstance(intervals, list) and isinstance(intervals[0], list):
        for inter in intervals:
            assert isinstance(inter, list) and len(inter) == 2 and isinstance(inter[0], (float, int)) and isinstance(
                inter[1], (float, int)), 'intervals must be list of 2 real numbers or list of lists of 2 real numbers'

    else:
        assert isinstance(intervals, list) and len(intervals) == 2 and isinstance(intervals[0],
                                                                                  (float, int)) and isinstance(
            intervals[1], (float, int)), 'intervals must be list of 2 real numbers or list of lists of 2 real numbers'

    assert type_remv in ["Zero", "NaN"], "the removing methode is not recognized"

    # Region Removal ---------------------------------------------------------------
    #get the remove interval based on the type of data
    if type_of_spec == 'serum':
        removed_inter = [4.5, 5.1]
    elif type_of_spec == 'urine':
        removed_inter = [4.5, 6.1]
    else:
        removed_inter = intervals
    # save the ppm mesurement that will not be reved into list
    list_of_pp_to_keep = []
    #get the ppm scale from the columns
    ppm = spectrum.columns

    if isinstance(removed_inter, list) and isinstance(intervals[0], list):
        for inter in removed_inter:
            new_list = []
            for i in ppm:
                if i < np.min(inter) or i > np.max(inter):
                    new_list.append(i)
            list_of_pp_to_keep = new_list.copy()
            ppm = list_of_pp_to_keep.copy()


    else:
        ppm = spectrum.columns
        for i in ppm:
            if i < np.min(removed_inter) or i > np.max(removed_inter):
                list_of_pp_to_keep.append(i)
    # get ht lsit of ppm mesuremnt to be removed
    list_remov = list(set(spectrum.columns) - set(list_of_pp_to_keep))
    #get a copy of data
    new_data = spectrum.copy()
    # replace the mesurement to be removed by zero
    if type_remv == "Zero":
        new_data[list_remov] = 0
    else:
        # replace the mesurement to be removed by Nan
        new_data[list_remov] = np.NaN
    if verbose == True:
        print('region removed: ', removed_inter)
        print('replaced with : ', type_remv)

    return new_data





