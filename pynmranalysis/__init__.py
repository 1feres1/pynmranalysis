from .analysis import PyPLS , PyPLS_DA , PyPCA

__version__ = '0.13.5'

__all__ = ['PyPLS' , 'PyPLS_DA', 'PyPCA']

"""
pynmranalysis provide objects witch wrap pre-existing scikit-learn PCA and PLS algorithms and add PLS-DA models
 with are very useful for dealing with omics dataset in general and NMR data specifically

ChemometricsPCA - PCA analysis object.
ChemometricsPLS - Object for Partial Least Squares regression analysis and regression quality metrics.
Chemometrics PLSDA - Object for Partial Least Squares Discriminant analysis (PLS followed by transformation of the 
Y prediction into class membership). Supports both 1vs1 and Multinomial classification schemes (although ROC curves
and quality control metrics for Multinomial are still work in progress).

"""