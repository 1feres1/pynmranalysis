from scipy import integrate
import pandas as pd
import numpy as np


def binning(spectrum, width=False, mb=500, bin_size=0.04, int_meth="simps", verbose=False):
    """
        , returns binned spectrum with every spectrum bins is the integral of the original spectrum along hthe
        ppm range

        :param: spectrum: dataframe of the spectrun values index are the cases and columns are the ppm values
        :param width: if width is True we use the bin size as a ppm window for binning
        :param bin_size: bin size is the ppm range of each bin
        :param int_meth: if set to simps use the composite Simpson’s rule for integration and trap use the composite trapezoidal rule.
        :param verbose: display inforamation about binning

        :return: binned spectrum with columns the bins range

        """
    # Data initialisation and checks ----------------------------------------------

    assert isinstance(width, bool), "the width arg must be True or False "
    assert mb > 0 and isinstance(mb, int), " mb must be positive integer "
    assert int_meth in ["simps", "traps"], "integration methode is indifined"
    assert isinstance(verbose, bool), "verbose must be boolean"

    # Bucketting ----------------------------------------------

    ppm = spectrum.columns
    if width == True:
        nb_bins = int((max(ppm) - min(ppm)) / bin_size)
    else:
        nb_bins = mb

    new_df = pd.DataFrame()

    buckets = np.array_split(spectrum.columns, nb_bins)
    for i, bucket in enumerate(buckets):
        if int_meth == "simps":
            x = integrate.simps(spectrum.loc[:, bucket], bucket[::-1])
        else:
            x = integrate.trapz(spectrum.loc[:, bucket], bucket[::-1])

        new_df[np.round(bucket[0], 3)] = x

    new_df.index = spectrum.index
    if verbose == True:
        print('number of bins : ', nb_bins)
    return new_df


def region_removal(spectrum, type_of_spec="manual", type_remv="Zero", intervals=[4.5, 5.1], verbose=False):
    """
          Removes the non-informative regions by setting the values of the spectra in these intervals to zero.

          :param: spectrum: dataframe of the spectrun values index are the cases and columns are the ppm values
          :param type_of_spec: if not "manual", will automatically remove unwanted regions depending on the nature of spectra.
          :param type_remv:  If equal to "zero", intensities are set to 0; if type.rr = "NaN", intensities are set to NaN.
          :param verbose: If "True", will print processing information.
          :params intervals: List containing the extremities of the intervals to be removed.
          :return: spectrum without the unwanted regions

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
    if type_of_spec == 'serum':
        removed_inter = [4.5, 5.1]
    elif type_of_spec == 'urine':
        removed_inter = [4.5, 6.1]
    else:
        removed_inter = intervals

    list_of_pp_to_keep = []
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

    list_remov = list(set(spectrum.columns) - set(list_of_pp_to_keep))
    new_data = spectrum.copy()
    if type_remv == "Zero":
        new_data[list_remov] = 0
    else:
        new_data[list_remov] = np.NaN
    if verbose == True:
        print('region removed: ', removed_inter)
        print('replaced with : ', type_remv)

    return new_data


