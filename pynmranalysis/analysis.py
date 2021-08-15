# import
import numpy as np

from sklearn.base import TransformerMixin , BaseEstimator , ClassifierMixin , clone ,RegressorMixin
from sklearn import metrics
from numpy import interp
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import f




#####################################################################################





class PyPCA(BaseEstimator):
    """
         PyPAC object - Wrapper for sklearn.decomposition PCA algorithms for Omics data analysis
        :param n_comps: Number of components for PCA
        :param scaler: data scaling object

    """

    def __init__(self, n_comps=2, scaler=StandardScaler()):

        # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
        pca = PCA(n_components=n_comps)
        assert isinstance(scaler,
                          TransformerMixin) or scaler is None, "sclaler must be an sklearn transformer-like or None"

        # initialize variabels
        self.pca_algorithm = pca
        self.n_comps = n_comps
        self.scaler = scaler
        self.loadings = None
        self.isfitted = False
        self.scores = None
        self.m_params = None

    def transform(self, x):
        """
        get the projection of the data metrix x on the pricipal componants of PCA
        :param x: data metrix to be fit (rows : samples , columns : variables )

        :return: PCA projections (x scores) (rows : samples , columns : principal componants)

        :raise ValueError: If there are problems with the input or during model fitting.
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.transform(xscaled)
            else:
                return self.pca_algorithm.transform(x)
        except ValueError as ver:
            raise ver

    def _residual_ssx(self, x):
        """
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :return: RSS resudual sum of squares
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)
        residuals = np.sum((xscaled - x_reconstructed) ** 2, axis=1)
        return residuals

    def inverse_transform(self, scores):
        """
        inverse transformation of x score data to the original data before projection
        :param scores: The projections ( x scores)  (rows : samples , columns : principal componants)

        :return: Data matrix in the original format (rows : samples , columns : variables )

        """
        # Scaling check for consistency
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    def fit_transform(self, x, **fit_params):
        """
        Fit a model and return the x scores (rows : samples , columns : principal componants)
        :param x:  data metrix to be fit (rows : samples , columns : variables )
        :return: PCA projections ( x scores) after transforming x
        :raise ValueError: If there are problems with the input or during model fitting.
        """

        try:
            self.fit(x, )
            return self.transform(x)
        except ValueError as ver:
            raise ver

    def fit(self, x):
        """
              Perform model fitting on the provided x data matrix and calculate basic goodness-of-fit metrics.
              :param x: data metrix to be fit (rows : samples , columns : variables )
              :raise ValueError: If any problem occurs during fitting.
        """
        try:
            # check if we will use scaling or not for PCA
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled)
                self.scores = self.pca_algorithm.transform(xscaled)
                ss = np.sum((xscaled - np.mean(xscaled, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((xscaled - predicted) ** 2)

            else:
                self.pca_algorithm.fit(x, )
                self.scores = self.pca_algorithm.transform(x)
                ss = np.sum((x - np.mean(x, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((x - predicted) ** 2)
            # set model parmetres
            self.m_params = {'R2X': 1 - (rss / ss), 'VarExp': self.pca_algorithm.explained_variance_,
                             'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}

            # For "Normalised" DmodX calculation
            resid_ssx = self._residual_ssx(x)
            s0 = np.sqrt(resid_ssx.sum() / ((self.scores.shape[0] - self.n_comps - 1) * (x.shape[1] - self.n_comps)))
            self.m_params['S0'] = s0

            # set loadings
            self.loadings = self.pca_algorithm.components_
            # set fitted to true
            self.isfitted = True


        except ValueError as ver:
            raise ver

    def hotelling_T2(self, comps=None, alpha=0.05):
        """
        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.
        :param list comps:
        :param float alpha: Significance level
        :return: The Hotelling T2 ellipsoid radii at vertex
        :raise AtributeError: If the model is not fitted
        :raise ValueError: If the components requested are higher than the number of components in the model
        :raise TypeError: If comps is not None or list/numpy 1d array and alpha a float
        """

        try:
            if self.isfitted is False:
                raise AttributeError("Model is not fitted yet ")
            n_samples = self.scores.shape[0]
            if comps is None:
                n_comps = self.n_comps
                ellips = self.scores[:, range(self.n_comps)] ** 2
                ellips = 1 / n_samples * (ellips.sum(0))
            else:
                n_comps = len(comps)
                ellips = self.scores[:, comps] ** 2
                ellips = 1 / n_samples * (ellips.sum(0))

            # F stat
            fs = (n_samples - 1) / n_samples * n_comps * (n_samples ** 2 - 1) / (n_samples * (n_samples - n_comps))
            fs = fs * f.ppf(1 - alpha, n_comps, n_samples - n_comps)

            hoteling_t2 = list()
            for comp in range(n_comps):
                hoteling_t2.append(np.sqrt((fs * ellips[comp])))

            return np.array(hoteling_t2)

        except AttributeError as atrer:
            raise atrer
        except ValueError as valer:
            raise valer
        except TypeError as typer:
            raise typer

    def dmodx(self, x):
        """
        Normalised DmodX measure
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :return: The Normalised DmodX measure for each sample
        """
        resids_ssx = self._residual_ssx(x)
        s = np.sqrt(resids_ssx / (self.loadings.shape[1] - self.n_comps))
        dmodx = np.sqrt((s / self.m_params['S0']) ** 2)
        return dmodx

    def _dmodx_fcrit(self, x, alpha=0.05):
        """
        :param alpha: significance level
        :return dmodx fcrit
        """

        # Degrees of freedom for the PCA model (denominator in F-stat)

        dmodx_fcrit = f.ppf(1 - alpha, x.shape[1] - self.n_comps - 1,
                            (x.shape[0] - self.n_comps - 1) * (x.shape[1] - self.n_comps))

        return dmodx_fcrit

    def outlier(self, x, comps=None, measure='T2', alpha=0.05):
        """
         using F statistic and T2 /Dmodx mesure to determine outliers
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param comps: Which components to use (for Hotelling T2 only)
        :param measure: T2 or DmodX
        :param alpha: Significance level
        :return: List of ouliers indices
        """
        try:
            if measure == 'T2':
                scores = self.transform(x)
                t2 = self.hotelling_T2(comps=comps)
                outlier_idx = np.where(((scores ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
            elif measure == 'DmodX':
                dmodx = self.dmodx(x)
                dcrit = self._dmodx_fcrit(x, alpha)
                outlier_idx = np.where(dmodx > dcrit)[0]
            else:
                print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
            return outlier_idx
        except Exception as exp:
            raise exp

    def score_plot(self):
        """ plot the projection of the x scores on the firest 2 components
        :param x : data metrix to be fit (rows : samples , columns : variables )
        :return 2 dimentional scatter plot """

        try:
            if self.isfitted == False:
                raise AttributeError("Model is not fitted yet ")

            plt.scatter(self.scores[:, 0], self.scores[:, 1] , s=100, edgecolors='k',)

            for i in range(self.scores.shape[0]):
                plt.text(x=self.scores[i, 0] + 0.3, y=self.scores[i, 1] + 0.3, s=i + 1)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.title('PCA score plot')
            plt.show()
        except AttributeError as atter:
            raise atter
        except TypeError as typer:
            raise typer

    def scree_plot(self):
        """ plot the explained varianace of each componant in the PCA model
        :param x : data metrix to be fit (rows : samples , columns : variables )
        :return scree plot  """

        try:
            if self.isfitted == False:
                raise AttributeError("Model is not fitted yet ")
            features = ['PC ' + str(x) for x in range(1, self.n_comps + 1)]
            plt.bar(features, self.m_params['VarExpRatio'], color='black')

            plt.ylabel('variance %')
            plt.xlabel('PCA features')
            plt.xticks = features
            plt.title('Scree plot')
            plt.show()
        except AttributeError as atter:
            raise atter
        except TypeError as typer:
            raise typer

    def outlier_plot(self, x, comps=None, measure='T2', alpha=0.05):
        """ detect outlier in x metric based on their variance and plot them with different color
        :param x : data metrix to be fit (rows : samples , columns : variables )
        :param comps: Which components to use (for Hotelling T2 only)
        :param measure: T2 or DmodX
        :param alpha: Significance level

        :return scree plot  """

        try:
            if self.isfitted == False:
                raise AttributeError("Model is not fitted yet ")
            # get ouliers index
            outliers = self.outlier(x=x, comps=comps, measure=measure, alpha=alpha)
            # not outlier index
            not_outliers = [x for x in np.arange(self.scores.shape[0]) if x not in outliers]

            plt.scatter(self.scores[not_outliers, 0], self.scores[not_outliers, 1], color='black', label='not outlier' ,s=100, edgecolors='k',)
            plt.scatter(self.scores[outliers, 0], self.scores[outliers, 1], color='r', label='outlier' , s=100, edgecolors='k',)
            for i in range(self.scores.shape[0]):
                plt.text(x=self.scores[i, 0] + 0.3, y=self.scores[i, 1] + 0.3, s=i + 1)

            plt.ylabel('PCA 2')
            plt.xlabel('PCA 1')
            plt.legend()
            plt.title('outliers plot')
            plt.show()
        except AttributeError as atter:
            raise atter
        except TypeError as typer:
            raise typer

    def target_plot(self,  y):
        """ the same as score plot but instead but we add color to each sample based on their classe
        :param x : data metrix to be fit (rows : samples , columns : variables )
        :params y : target variable (list) (each class has unique integer value)

        :return scree plot  """
        assert isinstance(y, (list, np.ndarray)) and len(y) == self.scores.shape[0]
        try:
            if self.isfitted == False:
                raise AttributeError("Model is not fitted yet ")

            targets = np.unique(y)
            colors = ['r', 'g']
            for target, color in zip(targets, colors):
                indicesToKeep = [x for x in np.arange(self.scores.shape[0]) if y[x] == target]

                plt.scatter(self.scores[indicesToKeep, 0]
                            , self.scores[indicesToKeep, 1]
                            , c=color, label='class ' + str(target) ,s=100, edgecolors='k',
                            )
            for i in range(self.scores.shape[0]):
                plt.text(x=self.scores[i, 0] + 0.3, y=self.scores[i, 1] + 0.3, s=i + 1)

            plt.ylabel('PCA 2')
            plt.xlabel('PCA 1')
            plt.legend()
            plt.title('target plot')
            plt.show()
        except AttributeError as atter:
            raise atter
        except TypeError as typer:
            raise typer


#########################################################################

class PyPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    PuPLS object semilar to PLS model in croos decomposition module in sklearn
    this object will be used for calculation of PLS-DA params (R2X , R2Y)
    :param int n_comps: number of componants for PLS model
    :param xscaler: Scaler object for X data matrix it should be an sklearn scaler or None
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.

    """
    """
    the implemntation of this algorithme is taken from the SIMPLS implementation 
    (SIMPLS, is proposed which calculates the PLS factors directly as linear combinations of the original variables)
    you can check the original paper "SIMPLS: An alternative approach to partial least squares regression", DOI : "https://doi.org/10.1016/0169-7439(93)85002-X"
    """

    def __init__(self, ncomps=2, xscaler=StandardScaler()):

        try:

            # call sklearn PLS model with the same number of componants
            pls_algorithm = PLSRegression(ncomps, scale=False)
            # verify that scaler used is sklearn based scaleer or None
            assert isinstance(xscaler, TransformerMixin) or xscaler is None, "scaler used is not defined"

            self.pls_algorithm = pls_algorithm
            # assign most variable to None because thay will be set when calling fit function
            self.scores_t = None  # projection of X
            self.scores_u = None  # projection of Y
            self.weights_w = None  # maximum covariance of X with Y
            self.weights_c = None  # maximum covariance
            self.loadings_p = None  # loading of model simelar to PCA loading assosiated with T to X
            self.loadings_q = None  # loading of model simelar to PCA loading assosiated with U to Y
            self.rotations_ws = None  # the rotation of X in the latin variable space
            self.rotations_cs = None  # the rotation of Y in the latin variable space
            self.b_u = None  # the beta from regration T on U
            self.b_t = None  # the beta from regression U on T
            self.beta_coeffs = None  # the cofficients of PLS regression model

            self.ncomps = ncomps  # number of component (altent variablels )
            self.x_scaler = xscaler  # scaler used on independent ariables X
            self.y_scaler = StandardScaler(with_std=False)  # scaler used on dependent ariables Y
            self.cvParameters = None  # cross validation params
            self.m_params = None  # model params
            self.isfitted = False  # boolien variable to indicate that model is fitted

        except TypeError as terr:
            print(terr.args[0])

    def fit(self, x, y):
        """
        fit the model to get all model coff and scores  and get the goodness of the fit R2X and R2Y
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        :raise ValueError: If any problem occurs during fitting.
        """
        try:
            x = np.array(x)
            # reshape Y by addding extra dimentien (requiremnt for PLS regression fitting )
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # reshape Y by addding extra dimentien (requiremnt for PLS regression fitting )
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            # scaler x if scaler is provided
            if self.x_scaler == None:
                xscaled = x
            else:
                xscaled = self.x_scaler.fit_transform(x)
                yscaled = self.y_scaler.fit_transform(y)

            # fit sklearn PLS regresion model to xscaled an y data
            self.pls_algorithm.fit(xscaled, yscaled)

            # Expose the model parameters
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # calclulate rotation from weights and loading
            self.rotations_cs = np.dot(np.linalg.pinv(np.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            # calculate beta from scores T and U
            self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_u.T, self.scores_u)), self.scores_u.T),
                              self.scores_t)
            self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_t.T, self.scores_t)), self.scores_t.T),
                              self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_
            # save that the model is fitted
            self.isfitted = True

            # get R2X and R2Y by calling score funtion
            R2Y = PyPLS.score(self, x=x, y=y, block_to_score='y')
            R2X = PyPLS.score(self, x=x, y=y, block_to_score='x')

            # get SSY SSX and composed SSX adn composed SSY
            cm_fit = self.cummulativefit(x, y)

            self.m_params = {'R2Y': R2Y, 'R2X': R2X, 'SSX': cm_fit['SSX'], 'SSY': cm_fit['SSY'],
                             'SSXcomp': cm_fit['SSXcomp'], 'SSYcomp': cm_fit['SSYcomp']}
            # calculate the sum of squares
            resid_ssx = self._residual_ssx(x)
            s0 = np.sqrt(resid_ssx.sum() / ((self.scores_t.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps)))
            self.m_params['S0X'] = s0

        except ValueError as verr:
            raise

    def score(self, x, y, block_to_score='y', sample_weight=None):
        """
        funtion to calculate R2X and R2Y
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: target variable
        :param str block_to_score: shose if we want to calculate R2X or R2Y
        :param sample_weight: Optional sample weights to use in scoring.
        :return R2Y: by predicting Y from X wr get R2Y
        :return R2X: by predicting X from Y we get R2X
        :raise ValueError: If block to score argument is not acceptable or date mismatch issues with the provided data.
        """
        try:
            if block_to_score not in ['x', 'y']:
                raise ValueError("x or y are the only accepted values for block_to_score")
            # reshape Y by addding extra dimentien (requiremnt for PLS regression fitting )
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # reshape Y by addding extra dimentien (requiremnt for PLS regression fitting )
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # calculate R2Y
            if block_to_score == 'y':
                yscaled = deepcopy(self.y_scaler).fit_transform(y)

                tssy = np.sum(np.square(yscaled))  # total sum of squares
                ypred = self.y_scaler.transform(PyPLS.predict(self, x, y=None))  # prediction of Y from X
                rssy = np.sum(np.square(yscaled - ypred))  # resudual sum of squres
                R2Y = 1 - (rssy / tssy)
                return R2Y
            # calculate R2X
            else:
                if self.x_scaler == None:
                    xscaled = x
                else:
                    xscaled = deepcopy(self.x_scaler).fit_transform(x)  # scale X
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                xpred = self.x_scaler.transform(PyPLS.predict(self, x=None, y=y))

                tssx = np.sum(np.square(xscaled))  # total sum of squres
                rssx = np.sum(np.square(xscaled - xpred))  # resuadual sum of squares "
                R2X = 1 - (rssx / tssx)
                return R2X
        except ValueError as verr:
            raise verr

    def predict(self, x=None, y=None):
        """
        predict y from X or X from Y
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: target variable
        :return: Predicted data block (X or Y) obtained from the other data block.
        :raise ValueError: If no data matrix is passed, or dimensions mismatch issues with the provided data.
        :raise AttributeError: Calling the method without fitting the model before.
        """

        try:
            # check if the odel is fitted or not
            if self.isfitted is True:
                if (x is not None) and (y is not None):
                    raise ValueError('target variable or predictive variable must be None ')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ('both predictive and target variable are None ')
                # Predict Y from X
                elif x is not None:
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    # sclae X if sclaer is provided
                    if self.x_scaler == None:
                        xscaled = x
                    else:
                        xscaled = self.x_scaler.fit_transform(x)
                    # Using Betas to predict Y directly
                    predicted = np.dot(xscaled, self.beta_coeffs)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.y_scaler.inverse_transform(predicted)

                    return predicted
                # Predict X from Y
                elif y is not None:
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = PyPLS.transform(self, x=None, y=y)

                    predicted = np.dot(np.dot(u_scores, self.b_u), self.weights_w.T)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.x_scaler.inverse_transform(predicted)
                    return predicted
            else:
                raise AttributeError("Model is not fitted")
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def transform(self, x=None, y=None):
        """
        calculate U or T metrix equivalent to sklearn TransformeMixin
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: target variable
        :return: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        :raise ValueError: If dimensions of input data are mismatched.
        :raise AttributeError: When calling the method before the model is fitted.
        """

        try:
            # Check if model is fitted
            if self.isfitted is True:
                if (x is not None) and (y is not None):
                    raise ValueError('target variable or predictive variable must be None ')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ('both predictive and target variable are None ')
                # If Y is given, return U
                elif x is None:
                    # reshape y by adding extra dimetion if y is a vector
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    yscaled = self.y_scaler.transform(y)
                    U = np.dot(yscaled, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # reshape x by adding extra dimention if its a vector
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)

                    xscaled = self.x_scaler.transform(x)
                    T = np.dot(xscaled, self.rotations_ws)
                    return T
            else:
                raise AttributeError('Model not fitted')

        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def cummulativefit(self, x, y):
        """
        calculate the commitative sum of squares
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: target variable
        :return: dictionary object containing the total Regression Sum of Squares and the Sum of Squares
        per components, for both the X and Y data blocks.
        """
        # reshape y if number of dimention is 1
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # reshapeX if number of dimention is 1
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # verifiy the model is fitted or not
        if self.isfitted is False:
            raise AttributeError('fit model first')

        if self.x_scaler == None:
            xscaled = x
        else:
            xscaled = self.x_scaler.fit_transform(x)
        yscaled = self.y_scaler.transform(y)

        # Obtain residual sum of squares for whole data set and per component
        SSX = np.sum(np.square(xscaled))  # sum of squres of predictve variables
        SSY = np.sum(np.square(yscaled))  # sum of squares of target variable
        ssx_comp = list()
        ssy_comp = list()
        # calculate sum of squres for each component
        for curr_comp in range(1, self.ncomps + 1):
            model = self._reduce_ncomps(curr_comp)

            ypred = PyPLS.predict(model, x, y=None)
            xpred = self.x_scaler.transform(PyPLS.predict(model, x=None, y=y))
            rssy = np.sum(np.square(y - ypred))
            rssx = np.sum(np.square(xscaled - xpred))
            ssx_comp.append(rssx)
            ssy_comp.append(rssy)
        # save the result
        cumulative_fit = {'SSX': SSX, 'SSY': SSY, 'SSXcomp': np.array(ssx_comp), 'SSYcomp': np.array(ssy_comp)}

        return cumulative_fit

    def _reduce_ncomps(self, n__comps):
        """
        get a semilar model with reduced number of componants
        :param int n__comps: number of componants
        :return PyPLS object with reduced number of components.
        :raise ValueError: If number of components desired is larger than original number of components
        :raise AttributeError: If model is not fitted.
        """
        try:
            # raise error if number of componat of the new model is bigger that the original
            if n__comps > self.ncomps:
                raise ValueError('Fit a new model with more components instead')
            # verify that the model is fitted or not
            if self.isfitted is False:
                raise AttributeError('Model not Fitted')
            # get the new model variable
            newmodel = deepcopy(self)
            newmodel.ncomps = n__comps

            newmodel.modelParameters = None
            newmodel.cvParameters = None
            newmodel.loadings_p = self.loadings_p[:, 0:n__comps]
            newmodel.weights_w = self.weights_w[:, 0:n__comps]
            newmodel.weights_c = self.weights_c[:, 0:n__comps]
            newmodel.loadings_q = self.loadings_q[:, 0:n__comps]
            newmodel.rotations_ws = self.rotations_ws[:, 0:n__comps]
            newmodel.rotations_cs = self.rotations_cs[:, 0:n__comps]
            newmodel.scores_t = None
            newmodel.scores_u = None
            newmodel.b_t = self.b_t[0:n__comps, 0:n__comps]
            newmodel.b_u = self.b_u[0:n__comps, 0:n__comps]

            # These have to be recalculated from the rotations
            newmodel.beta_coeffs = np.dot(newmodel.rotations_ws, newmodel.loadings_q.T)

            return newmodel
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def inverse_transform(self, t=None, u=None):
        """
        inverse transorm of the : genrate x and y from U and T scores
        :param t: T scores corresponding to the X data matrix.
        :param u: Y scores corresponding to the Y data vector/matrix.
        :return x: data metrix to be fit (rows : samples , columns : variables )
        :return y: target variable
        :rtype: numpy.ndarray, shape [n_samples, n_features] or None
        :raise ValueError: If dimensions of input data are mismatched.
        """
        try:
            # check if the model is fitted o not
            if self.isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('u or t must be None')
                elif t is None and u is None:
                    raise ValueError('both variable are None ')
                elif t is not None:
                    # get the prdcition from t and the transpose of p_loadings
                    xpred = np.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled
                # If U is given, return T
                elif u is not None:
                    # get the prediction from u and q loadings transpose
                    ypred = np.dot(u, self.loadings_q.T)

                    yscaled = self.y_scaler.inverse_transform(ypred)

                    return yscaled

        except ValueError as verr:
            raise verr

    def _residual_ssx(self, x):
        """
        calculate the resudua sum of squres
        :param x: Data matrix [n samples, m variables]
        :return: The residual Sum of Squares per sample
        """
        # transorm x to laten variables
        pred_scores = self.transform(x)
        # transform the latent variables back to original space
        x_reconstructed = self.x_scaler.transform(self.inverse_transform(pred_scores))
        # scale x if scaler is provided
        if self.x_scaler == None:
            xscaled = x
        else:
            xscaled = self.x_scaler.fit_transform(x)
        # calculate the resudual
        residuals = np.sum(np.square(xscaled - x_reconstructed), axis=1)
        return residuals


############################################################

class PyPLS_DA(PyPLS, ClassifierMixin):
    """
    PyPLS_DA object -Function to perform standard Partial Least Squares regression to classify samples.

    :param int ncomps: Number of components for PLS-DA model
    :param xscaler: Scaler object for X data matrix.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    """
    plsda function fit PLS models with 1,...,ncomp components to the factor or class vector Y. The appropriate indicator matrix is created.

    standar scaler or any other scaling technqiue is applyed  as internal pre-processing step
    See:
    - Indhal et. al., From dummy regression to prior probabilities in PLS-DA, Journal of Chemometrics, 2007
    - Barker, Matthew, Rayens, William, Partial least squares for discrimination, Journal of Chemometrics, 2003
    - Brereton, Richard G. Lloyd, Gavin R., Partial least squares discriminant analysis: Taking the magic away, 
    Journal of Chemometrics, 2014

    Model performance metrics employed are the Q2Y , Area under the curve and ROC curves, f1 measure, balanced accuracy,
    precision, recall, confusion matrices and 0-1 loss.
    """

    def __init__(self, ncomps=2, xscaler=StandardScaler()):
        pls_algorithm = PLSRegression(ncomps, scale=False)
        try:

            # chek if the providede scaler is sklearn scaler or not
            assert isinstance(xscaler,
                              TransformerMixin) or xscaler is None, "sclaler must be an sklearn transformer-like or None"

            self.pls_algorithm = pls_algorithm

            # set model variable most of them are set now to None because they get change when fitting the model
            self.scores_t = None  # projection of X
            self.scores_u = None  # projection of Y
            self.weights_w = None  # maximum covariance of X with Y
            self.weights_c = None  # maximum covariance
            self.loadings_p = None  # loading of model simelar to PCA loading assosiated with T to X
            self.loadings_q = None  # loading of model simelar to PCA loading assosiated with U to Y
            self.rotations_ws = None  # the rotation of X in the latin variable space
            self.rotations_cs = None  # the rotation of Y in the latin variable space
            self.b_u = None  # the beta from regration T on U
            self.b_t = None  # the beta from regression U on T
            self.beta_coeffs = None  # the cofficients of PLS regression model
            self.n_classes = None  # number of distanct classes in target variable
            self.class_means = None
            self.ncomps = ncomps  # number of component (altent variablels )
            self.x_scaler = xscaler  # scaler used on independent ariables X
            self.y_scaler = StandardScaler(with_std=False)  # scaler used on dependent ariables Y
            self.cvParameters = None  # cross validation params
            self.m_params = None  # model params
            self.isfitted = False  # boolien variable to indicate that model is fitted
        except TypeError as ter:
            print(ter.args[0])
            raise ter

    def fit(self, x, y, ):
        """
        Fit model to data (x and y)
        :param x:array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        :param y: array-like of shape (n_samples,)
            Target vectors, where `n_samples` is the number of samples

        :raise ValueError: If any problem occurs during fitting.
        """
        try:

            # reshape x if number of dimentions equal to 1 by adding extra dimention
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            # if scaler is not None scale x
            if self.x_scaler == None:
                xscaled = x
            else:
                xscaled = self.x_scaler.fit_transform(x)

                # get the nymber of classes
            n_classes = np.unique(y).size
            self.n_classes = n_classes

            # create a dummy metrix if number of classes exited 2
            if self.n_classes > 2:
                dummy_mat = pd.get_dummies(y).values
                y_scaled = self.y_scaler.fit_transform(dummy_mat)
            else:
                # if number of dimensions equal to 1 add extra dim
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                y_scaled = self.y_scaler.fit_transform(y)

            # fit PLS regression model
            self.pls_algorithm.fit(xscaled, y_scaled)

            # get the model params from the fitted PLS model
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # calculate rotation and beta variable using loading and weight of PLS model
            self.rotations_cs = np.dot(np.linalg.pinv(np.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_u.T, self.scores_u)), self.scores_u.T),
                              self.scores_t)
            self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_t.T, self.scores_t)), self.scores_t.T),
                              self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_
            # create class mean matrix based on obtained  T score
            self.class_means = np.zeros((n_classes, self.ncomps))
            for curr_class in range(self.n_classes):
                curr_class_idx = np.where(y == curr_class)
                self.class_means[curr_class, :] = np.mean(self.scores_t[curr_class_idx])

            # save that the model is fitted
            self.isfitted = True

            # calculate R2X and R2Y in both cases binery and non binery classification
            if self.n_classes > 2:
                R2Y = PyPLS.score(self, x=x, y=dummy_mat, block_to_score='y')
                R2X = PyPLS.score(self, x=x, y=dummy_mat, block_to_score='x')
            else:
                R2Y = PyPLS.score(self, x=x, y=y, block_to_score='y')
                R2X = PyPLS.score(self, x=x, y=y, block_to_score='x')

            # constant grid for ROC
            fpr_grid = np.linspace(0, 1, num=20)

            # get class scores
            class_score = PyPLS.predict(self, x=x)
            # binery classification
            if n_classes == 2:
                y_pred = self.predict(x)
                accuracy = metrics.accuracy_score(y, y_pred)
                precision = metrics.precision_score(y, y_pred)
                recall = metrics.recall_score(y, y_pred)
                misclassified_samples = np.where(y.ravel() != y_pred.ravel())[0]
                f1_score = metrics.f1_score(y, y_pred)
                conf_matrix = metrics.confusion_matrix(y, y_pred)
                zero_oneloss = metrics.zero_one_loss(y, y_pred)
                matthews_mcc = metrics.matthews_corrcoef(y, y_pred)

                # Interpolated ROC curve and AUC
                roc_curve = metrics.roc_curve(y, class_score.ravel())
                tpr = roc_curve[1]
                fpr = roc_curve[0]
                interpolated_tpr = np.zeros_like(fpr_grid)
                interpolated_tpr += interp(fpr_grid, fpr, tpr)
                roc_curve = (fpr_grid, interpolated_tpr, roc_curve[2])
                auc_area = metrics.auc(fpr_grid, interpolated_tpr)

            else:
                # multi class classification
                y_pred = self.predict(x)
                accuracy = metrics.accuracy_score(y, y_pred)
                precision = metrics.precision_score(y, y_pred, average='weighted')
                recall = metrics.recall_score(y, y_pred, average='weighted')
                misclassified_samples = np.where(y.ravel() != y_pred.ravel())[0]
                f1_score = metrics.f1_score(y, y_pred, average='weighted')
                conf_matrix = metrics.confusion_matrix(y, y_pred)
                zero_oneloss = metrics.zero_one_loss(y, y_pred)
                matthews_mcc = np.nan
                roc_curve = list()
                auc_area = list()

                # Generate multiple ROC curves - one for each class the multiple class case
                for predclass in range(self.n_classes):
                    current_roc = metrics.roc_curve(y, class_score[:, predclass], pos_label=predclass)
                    # Interpolate all ROC curves to a finite grid
                    #  Makes it easier to average and compare multiple models - with CV in mind
                    tpr = current_roc[1]
                    fpr = current_roc[0]

                    interpolated_tpr = np.zeros_like(fpr_grid)
                    interpolated_tpr += interp(fpr_grid, fpr, tpr)
                    roc_curve.append([fpr_grid, interpolated_tpr, current_roc[2]])
                    auc_area.append(metrics.auc(fpr_grid, interpolated_tpr))

            # Obtain residual sum of squares for whole data set and per component
            # Same as Chemometrics PLS, this is so we can use VIP's and other metrics as usual
            if self.n_classes > 2:
                cm_fit = self.cummulativefit(x, dummy_mat)
            else:
                cm_fit = self.cummulativefit(x, y)

            # save the model params
            self.m_params = {'PLS': {'R2Y': R2Y, 'R2X': R2X, 'SSX': cm_fit['SSX'], 'SSY': cm_fit['SSY'],
                                     'SSXcomp': cm_fit['SSXcomp'], 'SSYcomp': cm_fit['SSYcomp']},
                             'DA': {'Accuracy': accuracy, 'AUC': auc_area,
                                    'ConfusionMatrix': conf_matrix, 'ROC': roc_curve,
                                    'MisclassifiedSamples': misclassified_samples,
                                    'Precision': precision, 'Recall': recall,
                                    'F1': f1_score, '0-1Loss': zero_oneloss, 'MatthewsMCC': matthews_mcc,
                                    'ClassPredictions': y_pred}}

        except ValueError as verr:
            raise verr

    def score(self, x, y, sample_weight=None):
        """
        Predict and calculate the R2 for the model using one of the data blocks (X or Y) provided.
        Equivalent to the scikit-learn ClassifierMixin score method.
        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features] or None
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features] or None
        :param str block_to_score: Which of the data blocks (X or Y) to calculate the R2 goodness of fit.
        :param sample_weight: Optional sample weights to use in scoring.
        :type sample_weight: numpy.ndarray, shape [n_samples] or None
        :return R2Y: The model's R2Y, calculated by predicting Y from X and scoring.
        :rtype: float
        :return R2X: The model's R2X, calculated by predicting X from Y and scoring.
        :rtype: float
        :raise ValueError: If block to score argument is not acceptable or date mismatch issues with the provided data.
        """
        try:
            # return metrics.accuracy_score(y, self.predict(x), sample_weight=sample_weight)
            return PyPLS.score(self, x, y, block_to_score='x')
        except ValueError as verr:
            raise verr

    def predict(self, x):
        """
        predict the value of the target variable based on predictive variable x
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        :return: Predicted data block Y by as discret values using argmin
        :raise ValueError: If no data matrix is passed, or dimensions mismatch issues with the provided data.
        :raise AttributeError: Calling the method without fitting the model before.
        """

        try:
            if self.isfitted is False:
                raise AttributeError("Model is not fitted")

            # based on original encoding as 0, 1 (binery classification )
            if self.n_classes == 2:
                y_pred = PyPLS.predict(self, x)
                class_pred = np.argmin(np.abs(y_pred - np.array([0, 1])), axis=1)

            else:
                # multiclass classification
                pred_scores = self.transform(x=x)
                # encode the predicted variable
                closest_class_mean = lambda x: np.argmin(np.linalg.norm((x - self.class_means), axis=1))
                class_pred = np.apply_along_axis(closest_class_mean, axis=1, arr=pred_scores)
            return class_pred

        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def inverse_transform(self, t=None, u=None):
        """
        transform T and U scores to get X and Y
        :param t: T scores corresponding to the X data matrix.
        :param u: Y scores corresponding to the Y data vector/matrix.
        :return x: data metrix to be fit (rows : samples , columns : variables )
        :return y: depentent variable or target variable
        :raise ValueError: If dimensions of input data are mismatched.
        """
        try:
            if self.isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('T or U scores must be set to None  ')

                elif t is None and u is None:
                    raise ValueError('T and U cant be both None ')
                # If T is given, return U
                elif t is not None:
                    # calculate x prediction
                    xpred = np.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled

                # If U is given, return T

                elif u is not None:
                    # calculate y bases on loading transpose
                    ypred = np.dot(u, self.loadings_q.T)
                    return ypred

        except ValueError as verr:
            raise verr

    def transform(self, x=None, y=None):
        """
        calculate U or T metrix equivalent to sklearn TransformeMixin
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        :return: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        :raise ValueError: If dimensions of input data are mismatched.
        :raise AttributeError: When calling the method before the model is fitted.
        """

        try:
            # Check if model is fitted or not
            if self.isfitted is True:
                # If X and Y are passed, complain and do nothing
                if (x is not None) and (y is not None):
                    raise ValueError('one of the variable must be None')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('both variables are set to None')
                # If Y is given, return U
                elif x is None:
                    # verify that y is a single vector
                    if y.ndim != 1:
                        raise TypeError('Please supply a dummy vector with integer as class membership')

                    # muticlass classification
                    if self.n_classes > 2:
                        y = self.y_scaler.transform(pd.get_dummies(y).values)
                    else:
                        # binery classification
                        if y.ndim == 1:
                            y = y.reshape(-1, 1)
                            y = self.y_scaler.transform(y)

                    U = np.dot(y, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # add extra dimention to x if its a vector
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    if self.x_scaler == None:
                        xscaled = x
                    else:
                        xscaled = self.x_scaler.fit_transform(x)

                    T = np.dot(xscaled, self.rotations_ws)
                    return T
            else:
                raise AttributeError('Model not fitted yet ')
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def cross_validation(self, x, y, cv_method=KFold(7, shuffle=True), outputdist=False,
                         ):
        """
        cross validation result of the model and calculate Q2
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        :param cv_method:  cross valiation method
        :param bool outputdist: Output the whole distribution for. Useful when ShuffleSplit or CrossValidators other than KFold.
        :return: dict of cross validation scores
        :raise TypeError: If the cv_method passed is not a scikit-learn CrossValidator object.
        :raise ValueError: If the x and y data matrices are invalid.
        """

        try:

            # Check if global model is fitted... and if not, fit it using all of X
            if self.isfitted is False:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state of the object is not modified during
            # the cross_validation method call
            cv_pipeline = deepcopy(self)
            # Number of splits
            ncvrounds = cv_method.get_n_splits()

            # Number of classes to select tell binary from multi-class discrimination parameter calculation
            n_classes = np.unique(y).size

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            # The y variable expected is a single vector with ints as class label - binary
            # and multiclass classification are allowed but not multilabel so this will work.
            # but for the PLS part in case of more than 2 classes a dummy matrix is constructed and kept separately
            # throughout
            if y.ndim == 1:
                # y = y.reshape(-1, 1)
                if self.n_classes > 2:
                    y_pls = pd.get_dummies(y).values
                    y_nvars = y_pls.shape[1]
                else:
                    y_nvars = 1
                    y_pls = y
            else:
                raise TypeError('Please supply a dummy vector with integer as class membership')

            # Initialize list structures to contain the fit
            cv_loadings_p = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_loadings_q = np.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_weights_w = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_weights_c = np.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_train_scores_t = list()
            cv_train_scores_u = list()

            # CV test scores more informative for ShuffleSplit than KFold but kept here anyway
            cv_test_scores_t = list()
            cv_test_scores_u = list()

            cv_rotations_ws = np.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_rotations_cs = np.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_betacoefs = np.zeros((ncvrounds, y_nvars, x_nvars))
            cv_vipsw = np.zeros((ncvrounds, x_nvars))

            cv_trainprecision = np.zeros(ncvrounds)
            cv_trainrecall = np.zeros(ncvrounds)
            cv_trainaccuracy = np.zeros(ncvrounds)
            cv_trainauc = np.zeros((ncvrounds, y_nvars))
            cv_trainmatthews_mcc = np.zeros(ncvrounds)
            cv_trainzerooneloss = np.zeros(ncvrounds)
            cv_trainf1 = np.zeros(ncvrounds)
            cv_trainclasspredictions = list()
            cv_trainroc_curve = list()
            cv_trainconfusionmatrix = list()
            cv_trainmisclassifiedsamples = list()

            cv_testprecision = np.zeros(ncvrounds)
            cv_testrecall = np.zeros(ncvrounds)
            cv_testaccuracy = np.zeros(ncvrounds)
            cv_testauc = np.zeros((ncvrounds, y_nvars))
            cv_testmatthews_mcc = np.zeros(ncvrounds)
            cv_testzerooneloss = np.zeros(ncvrounds)
            cv_testf1 = np.zeros(ncvrounds)
            cv_testclasspredictions = list()
            cv_testroc_curve = list()
            cv_testconfusionmatrix = list()
            cv_testmisclassifiedsamples = list()

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = np.sum(np.square(cv_pipeline.x_scaler.fit_transform(x)))
            ssy = np.sum(np.square(cv_pipeline.y_scaler.fit_transform(y_pls.reshape(-1, 1))))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = np.zeros(ncvrounds)
            R2Y_training = np.zeros(ncvrounds)
            # R2X and R2Y assessed in the test set
            R2X_test = np.zeros(ncvrounds)
            R2Y_test = np.zeros(ncvrounds)
            x = np.array(x)
            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                # split the data explicitly
                train = train_testidx[0]
                test = train_testidx[1]

                # Check dimensions for the indexing
                ytrain = y[train]
                ytest = y[test]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                cv_pipeline.fit(xtrain, ytrain)
                # Prepare the scaled X and Y test data

                # Comply with the sklearn scaler behaviour
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                xtest_scaled = cv_pipeline.x_scaler.transform(xtest)

                R2X_training[cvround] = PyPLS.score(cv_pipeline, xtrain, ytrain, 'x')
                R2Y_training[cvround] = PyPLS.score(cv_pipeline, xtrain, ytrain, 'y')

                if y_pls.ndim > 1:
                    yplstest = y_pls[test, :]

                else:
                    yplstest = y_pls[test].reshape(-1, 1)

                # Use super here  for Q2
                ypred = PyPLS.predict(cv_pipeline, x=xtest, y=None)
                xpred = PyPLS.predict(cv_pipeline, x=None, y=ytest)

                xpred = cv_pipeline.x_scaler.transform(xpred).squeeze()
                ypred = cv_pipeline.y_scaler.transform(ypred).squeeze()

                curr_pressx = np.sum(np.square(xtest_scaled - xpred))
                curr_pressy = np.sum(np.square(cv_pipeline.y_scaler.transform(yplstest).squeeze() - ypred))

                R2X_test[cvround] = PyPLS.score(cv_pipeline, xtest, yplstest, 'x')
                R2Y_test[cvround] = PyPLS.score(cv_pipeline, xtest, yplstest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                cv_loadings_p[cvround, :, :] = cv_pipeline.loadings_p
                cv_loadings_q[cvround, :, :] = cv_pipeline.loadings_q
                cv_weights_w[cvround, :, :] = cv_pipeline.weights_w
                cv_weights_c[cvround, :, :] = cv_pipeline.weights_c
                cv_rotations_ws[cvround, :, :] = cv_pipeline.rotations_ws
                cv_rotations_cs[cvround, :, :] = cv_pipeline.rotations_cs
                cv_betacoefs[cvround, :, :] = cv_pipeline.beta_coeffs.T
                cv_vipsw[cvround, :] = cv_pipeline.VIP()

                # Training metrics
                cv_trainaccuracy[cvround] = cv_pipeline.m_params['DA']['Accuracy']
                cv_trainprecision[cvround] = cv_pipeline.m_params['DA']['Precision']
                cv_trainrecall[cvround] = cv_pipeline.m_params['DA']['Recall']
                cv_trainauc[cvround, :] = cv_pipeline.m_params['DA']['AUC']
                cv_trainf1[cvround] = cv_pipeline.m_params['DA']['F1']
                cv_trainmatthews_mcc[cvround] = cv_pipeline.m_params['DA']['MatthewsMCC']
                cv_trainzerooneloss[cvround] = cv_pipeline.m_params['DA']['0-1Loss']

                # Check this indexes, same as CV scores
                cv_trainmisclassifiedsamples.append(
                    train[cv_pipeline.m_params['DA']['MisclassifiedSamples']])
                cv_trainclasspredictions.append(
                    [*zip(train, cv_pipeline.m_params['DA']['ClassPredictions'])])

                cv_trainroc_curve.append(cv_pipeline.m_params['DA']['ROC'])

                fpr_grid = np.linspace(0, 1, num=20)

                y_pred = cv_pipeline.predict(xtest)
                # Obtain the class score
                class_score = PyPLS.predict(cv_pipeline, xtest)

                if n_classes == 2:
                    test_accuracy = metrics.accuracy_score(ytest, y_pred)
                    test_precision = metrics.precision_score(ytest, y_pred)
                    test_recall = metrics.recall_score(ytest, y_pred)
                    test_f1_score = metrics.f1_score(ytest, y_pred)
                    test_zero_oneloss = metrics.zero_one_loss(ytest, y_pred)
                    test_matthews_mcc = metrics.matthews_corrcoef(ytest, y_pred)
                    test_roc_curve = metrics.roc_curve(ytest, class_score.ravel())

                    # Interpolated ROC curve and AUC
                    tpr = test_roc_curve[1]
                    fpr = test_roc_curve[0]
                    interpolated_tpr = np.zeros_like(fpr_grid)
                    interpolated_tpr += interp(fpr_grid, fpr, tpr)
                    test_roc_curve = (fpr_grid, interpolated_tpr, test_roc_curve[2])
                    test_auc_area = metrics.auc(fpr_grid, interpolated_tpr)

                else:
                    test_accuracy = metrics.accuracy_score(ytest, y_pred)
                    test_precision = metrics.precision_score(ytest, y_pred, average='weighted')
                    test_recall = metrics.recall_score(ytest, y_pred, average='weighted')
                    test_f1_score = metrics.f1_score(ytest, y_pred, average='weighted')
                    test_zero_oneloss = metrics.zero_one_loss(ytest, y_pred)
                    test_matthews_mcc = np.nan
                    test_roc_curve = list()
                    test_auc_area = list()
                    # Generate multiple ROC curves - one for each class the multiple class case
                    for predclass in range(cv_pipeline.n_classes):
                        roc_curve = metrics.roc_curve(ytest, class_score[:, predclass], pos_label=predclass)
                        # Interpolate all ROC curves to a finite grid
                        #  Makes it easier to average and compare multiple models - with CV in mind
                        tpr = roc_curve[1]
                        fpr = roc_curve[0]
                        interpolated_tpr = np.zeros_like(fpr_grid)
                        interpolated_tpr += interp(fpr_grid, fpr, tpr)
                        test_roc_curve.append(fpr_grid, interpolated_tpr, roc_curve[2])
                        test_auc_area.append(metrics.auc(fpr_grid, interpolated_tpr))

                # TODO check the roc curve in train and test set
                # Check the actual indexes in the original samples
                test_misclassified_samples = test[np.where(ytest.ravel() != y_pred.ravel())[0]]
                test_classpredictions = [*zip(test, y_pred)]
                test_conf_matrix = metrics.confusion_matrix(ytest, y_pred)

                # Test metrics
                cv_testaccuracy[cvround] = test_accuracy
                cv_testprecision[cvround] = test_precision
                cv_testrecall[cvround] = test_recall
                cv_testauc[cvround, :] = test_auc_area
                cv_testf1[cvround] = test_f1_score
                cv_testmatthews_mcc[cvround] = test_matthews_mcc
                cv_testzerooneloss[cvround] = test_zero_oneloss
                # Check this indexes, same as CV scores
                cv_testmisclassifiedsamples.append(test_misclassified_samples)
                cv_testroc_curve.append(test_roc_curve)
                cv_testconfusionmatrix.append(test_conf_matrix)
                cv_testclasspredictions.append(test_classpredictions)

            # Do a proper investigation on how to get CV scores decently
            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings from
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in the X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = np.argmin(
                        np.array([np.sum(np.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload])),
                                  np.sum(np.abs(
                                      self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload] * -1))]))
                    if choice == 1:
                        cv_loadings_p[cvround, :, currload] = -1 * cv_loadings_p[cvround, :, currload]
                        cv_loadings_q[cvround, :, currload] = -1 * cv_loadings_q[cvround, :, currload]
                        cv_weights_w[cvround, :, currload] = -1 * cv_weights_w[cvround, :, currload]
                        cv_weights_c[cvround, :, currload] = -1 * cv_weights_c[cvround, :, currload]
                        cv_rotations_ws[cvround, :, currload] = -1 * cv_rotations_ws[cvround, :, currload]
                        cv_rotations_cs[cvround, :, currload] = -1 * cv_rotations_cs[cvround, :, currload]
                        cv_train_scores_t.append([*zip(train, -1 * cv_pipeline.scores_t)])
                        cv_train_scores_u.append([*zip(train, -1 * cv_pipeline.scores_u)])
                        cv_test_scores_t.append([*zip(test, -1 * cv_pipeline.scores_t)])
                        cv_test_scores_u.append([*zip(test, -1 * cv_pipeline.scores_u)])
                    else:
                        cv_train_scores_t.append([*zip(train, cv_pipeline.scores_t)])
                        cv_train_scores_u.append([*zip(train, cv_pipeline.scores_u)])
                        cv_test_scores_t.append([*zip(test, cv_pipeline.scores_t)])
                        cv_test_scores_u.append([*zip(test, cv_pipeline.scores_u)])

            # Calculate Q-squareds
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.cvParameters = {'PLS': {'Q2X': q_squaredx, 'Q2Y': q_squaredy,
                                         'MeanR2X_Training': np.mean(R2X_training),
                                         'MeanR2Y_Training': np.mean(R2Y_training),
                                         'StdevR2X_Training': np.std(R2X_training),
                                         'StdevR2Y_Training': np.std(R2X_training),
                                         'MeanR2X_Test': np.mean(R2X_test),
                                         'MeanR2Y_Test': np.mean(R2Y_test),
                                         'StdevR2X_Test': np.std(R2X_test),
                                         'StdevR2Y_Test': np.std(R2Y_test)}, 'DA': {}}
            # Means and standard deviations...
            self.cvParameters['PLS']['Mean_Loadings_q'] = cv_loadings_q.mean(0)
            self.cvParameters['PLS']['Stdev_Loadings_q'] = cv_loadings_q.std(0)
            self.cvParameters['PLS']['Mean_Loadings_p'] = cv_loadings_p.mean(0)
            self.cvParameters['PLS']['Stdev_Loadings_p'] = cv_loadings_q.std(0)
            self.cvParameters['PLS']['Mean_Weights_c'] = cv_weights_c.mean(0)
            self.cvParameters['PLS']['Stdev_Weights_c'] = cv_weights_c.std(0)
            self.cvParameters['PLS']['Mean_Weights_w'] = cv_weights_w.mean(0)
            self.cvParameters['PLS']['Stdev_Weights_w'] = cv_weights_w.std(0)
            self.cvParameters['PLS']['Mean_Rotations_ws'] = cv_rotations_ws.mean(0)
            self.cvParameters['PLS']['Stdev_Rotations_ws'] = cv_rotations_ws.std(0)
            self.cvParameters['PLS']['Mean_Rotations_cs'] = cv_rotations_cs.mean(0)
            self.cvParameters['PLS']['Stdev_Rotations_cs'] = cv_rotations_cs.std(0)
            self.cvParameters['PLS']['Mean_Beta'] = cv_betacoefs.mean(0)
            self.cvParameters['PLS']['Stdev_Beta'] = cv_betacoefs.std(0)
            self.cvParameters['PLS']['Mean_VIP'] = cv_vipsw.mean(0)
            self.cvParameters['PLS']['Stdev_VIP'] = cv_vipsw.std(0)
            self.cvParameters['DA']['Mean_MCC'] = cv_testmatthews_mcc.mean(0)
            self.cvParameters['DA']['Stdev_MCC'] = cv_testmatthews_mcc.std(0)
            self.cvParameters['DA']['Mean_Recall'] = cv_testrecall.mean(0)
            self.cvParameters['DA']['Stdev_Recall'] = cv_testrecall.std(0)
            self.cvParameters['DA']['Mean_Precision'] = cv_testprecision.mean(0)
            self.cvParameters['DA']['Stdev_Precision'] = cv_testprecision.std(0)
            self.cvParameters['DA']['Mean_Accuracy'] = cv_testaccuracy.mean(0)
            self.cvParameters['DA']['Stdev_Accuracy'] = cv_testaccuracy.std(0)
            self.cvParameters['DA']['Mean_f1'] = cv_testf1.mean(0)
            self.cvParameters['DA']['Stdev_f1'] = cv_testf1.std(0)
            self.cvParameters['DA']['Mean_0-1Loss'] = cv_testzerooneloss.mean(0)
            self.cvParameters['DA']['Stdev_0-1Loss'] = cv_testzerooneloss.std(0)
            self.cvParameters['DA']['Mean_AUC'] = cv_testauc.mean(0)
            self.cvParameters['DA']['Stdev_AUC'] = cv_testauc.std(0)

            self.cvParameters['DA']['Mean_ROC'] = np.mean(np.array([x[1] for x in cv_testroc_curve]), axis=0)
            self.cvParameters['DA']['Stdev_ROC'] = np.std(np.array([x[1] for x in cv_testroc_curve]), axis=0)
            # TODO add cv scores averaging and stdev properly
            # Means and standard deviations...
            # self.cvParameters['Mean_Scores_t'] = cv_scores_t.mean(0)
            # self.cvParameters['Stdev_Scores_t'] = cv_scores_t.std(0)
            # self.cvParameters['Mean_Scores_u'] = cv_scores_u.mean(0)
            # self.cvParameters['Stdev_Scores_u'] = cv_scores_u.std(0)
            # Save everything found during CV
            if outputdist is True:
                self.cvParameters['PLS']['CVR2X_Training'] = R2X_training
                self.cvParameters['PLS']['CVR2Y_Training'] = R2Y_training
                self.cvParameters['PLS']['CVR2X_Test'] = R2X_test
                self.cvParameters['PLS']['CVR2Y_Test'] = R2Y_test
                self.cvParameters['PLS']['CV_Loadings_q'] = cv_loadings_q
                self.cvParameters['PLS']['CV_Loadings_p'] = cv_loadings_p
                self.cvParameters['PLS']['CV_Weights_c'] = cv_weights_c
                self.cvParameters['PLS']['CV_Weights_w'] = cv_weights_w
                self.cvParameters['PLS']['CV_Rotations_ws'] = cv_rotations_ws
                self.cvParameters['PLS']['CV_Rotations_cs'] = cv_rotations_cs
                self.cvParameters['PLS']['CV_TestScores_t'] = cv_test_scores_t
                self.cvParameters['PLS']['CV_TestScores_u'] = cv_test_scores_u
                self.cvParameters['PLS']['CV_TrainScores_t'] = cv_train_scores_t
                self.cvParameters['PLS']['CV_TrainScores_u'] = cv_train_scores_u
                self.cvParameters['PLS']['CV_Beta'] = cv_betacoefs
                self.cvParameters['PLS']['CV_VIPw'] = cv_vipsw

                # CV Test set metrics - The metrics which matter to benchmark classifier
                self.cvParameters['DA']['CV_TestMCC'] = cv_testmatthews_mcc
                self.cvParameters['DA']['CV_TestRecall'] = cv_testrecall
                self.cvParameters['DA']['CV_TestPrecision'] = cv_testprecision
                self.cvParameters['DA']['CV_TestAccuracy'] = cv_testaccuracy
                self.cvParameters['DA']['CV_Testf1'] = cv_testf1
                self.cvParameters['DA']['CV_Test0-1Loss'] = cv_testzerooneloss
                self.cvParameters['DA']['CV_TestROC'] = cv_testroc_curve
                self.cvParameters['DA']['CV_TestConfusionMatrix'] = cv_testconfusionmatrix
                self.cvParameters['DA']['CV_TestSamplePrediction'] = cv_testclasspredictions
                self.cvParameters['DA']['CV_TestMisclassifiedsamples'] = cv_testmisclassifiedsamples
                self.cvParameters['DA']['CV_TestAUC'] = cv_testauc
                # CV Train parameters - so we can keep a look on model performance in training set
                self.cvParameters['DA']['CV_TrainMCC'] = cv_trainmatthews_mcc
                self.cvParameters['DA']['CV_TrainRecall'] = cv_trainrecall
                self.cvParameters['DA']['CV_TrainPrecision'] = cv_trainprecision
                self.cvParameters['DA']['CV_TrainAccuracy'] = cv_trainaccuracy
                self.cvParameters['DA']['CV_Trainf1'] = cv_trainf1
                self.cvParameters['DA']['CV_Train0-1Loss'] = cv_trainzerooneloss
                self.cvParameters['DA']['CV_TrainROC'] = cv_trainroc_curve
                self.cvParameters['DA']['CV_TrainConfusionMatrix'] = cv_trainconfusionmatrix
                self.cvParameters['DA']['CV_TrainSamplePrediction'] = cv_trainclasspredictions
                self.cvParameters['DA']['CV_TrainMisclassifiedsamples'] = cv_trainmisclassifiedsamples
                self.cvParameters['DA']['CV_TrainAUC'] = cv_trainauc
            return None

        except TypeError as terp:
            raise terp

    def VIP(self, mode='w', direction='y'):
        """
        calculate the variable importance parameters to get the most important variable used by the model
        :param mode: The type of model parameter to use in calculating the VIP. Default value is weights (w), and other acceptable arguments are p, ws, cs, c and q.
        :type mode: str
        :param str direction: The data block to be used to calculated the model fit and regression sum of squares.
        :return numpy.ndarray VIP: The vector with the calculated VIP values.
        :rtype: numpy.ndarray, shape [n_features]
        :raise ValueError: If mode or direction is not a valid option.
        :raise AttributeError: Calling method without a fitted model.
        """
        try:
            # Code not really adequate for each Y variable in the multi-Y case - SSy should be changed so
            # that it is calculated for each y and not for the whole block
            if self.isfitted is False:
                raise AttributeError("Model is not fitted")
            if mode not in ['w', 'p', 'ws', 'cs', 'c', 'q']:
                raise ValueError("Invalid type of VIP coefficient")
            if direction not in ['x', 'y']:
                raise ValueError("direction must be x or y")

            choices = {'w': self.weights_w, 'p': self.loadings_p, 'ws': self.rotations_ws, 'cs': self.rotations_cs,
                       'c': self.weights_c, 'q': self.loadings_q}

            if direction == 'y':
                ss_dir = 'SSYcomp'
            else:
                ss_dir = 'SSXcomp'

            nvars = self.loadings_p.shape[0]
            vipnum = np.zeros(nvars)
            for comp in range(0, self.ncomps):
                vipnum += (choices[mode][:, comp] ** 2) * (self.m_params['PLS'][ss_dir][comp])

            vip = np.sqrt(vipnum * nvars / self.m_params['PLS'][ss_dir].sum())

            return vip

        except AttributeError as atter:
            raise atter
        except ValueError as verr:
            raise verr

    def permuation_test(x, y, nb_perm=20):
        """
        this function is still in developpement
        """
        return None

    def inertia_barplot(self, x, y):
        """
        interia plot to get the goodness of the fit R2 and the goodness of prediction Q2 with each number of componant
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        """
        Q2_scores = []
        R2_scores = []
        for i in range(1, self.ncomps + 1):
            # scores = cross_validate(pls_binary , X = df, y = Y1 ,scoring=scoring , cv=7 , n_jobs=-1 verbose=2 , ,return_train_score=True ,  )

            # create neww instance with diiferent number f componant
            plsda = PyPLS_DA(i)
            plsda.fit(x, y)
            plsda.cross_validation(x, y)
            R2_scores.append(plsda.m_params["PLS"]['R2Y'])
            Q2_scores.append(plsda.cvParameters['PLS']['Q2Y'])

        features = np.arange(len(Q2_scores))

        plt.bar(features - 0.2, R2_scores, 0.4, label='R2')
        plt.bar(features + 0.2, Q2_scores, 0.4, label='Q2')
        plt.legend()
        plt.title('interia plot')

    def score_plot(self, y):
        """
        PLS_DA sore plot gives the projection of the simples on the first 2 componants (latent variables )
        :param x: data metrix to be fit (rows : samples , columns : variables )
        :param y: depentent variable or target variable
        """
        try:
            if self.isfitted == False:
                raise AttributeError("Model is not fitted yet ")
            targets = np.unique(y)
            colors = ['r', 'g']
            for target, color in zip(targets, colors):
                indicesToKeep = [x for x in np.arange(self.scores_t.shape[0]) if y[x] == target]

                plt.scatter(self.scores_t[indicesToKeep, 0]
                            , self.scores_t[indicesToKeep, 1]
                            , c=color, label='class ' + str(target), s=100, edgecolors='k',
                            )
            for i in range(self.scores_t.shape[0]):
                plt.text(x=self.scores_t[i, 0] + 0.3, y=self.scores_t[i, 1] + 0.3, s=i + 1)

            plt.xlabel('LV 1')
            plt.ylabel('LV 2')
            plt.legend()
            plt.title('PLS-DA score plot')
            plt.show()
        except AttributeError as atter:
            raise atter
        except TypeError as typer:
            raise typer




