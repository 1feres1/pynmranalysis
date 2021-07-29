# import
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold , cross_validate
from sklearn.metrics import r2_score , make_scorer

def pyPCA(data, Y=None, standarize=True, n_components=10):
    """
         retrun the projection of the data on the specified PCA components with the score plot and projection plot on the first 2 principal componenets
         if y is not None each class will be colored diffrently

         :param: data: dataframe containg the independent variables
         :param Y: the dependent variable
         :param standarize if set to true independent variables will be automaticly standazized
         :return: scree plot (left fig) and xscore plot (right fig)

    """
    # Data initialisation and checks ----------------------------------------------

    assert isinstance(Y, list) or Y is None or type(Y).__module__ == np.__name__, "target data msut be a factor "
    assert isinstance(standarize, bool), 'standarize must be boolean instance'
    if Y is None:
        if standarize == True:
            # Standardize the data to have a mean of ~0 and a variance of 1
            X_std = StandardScaler().fit_transform(data)
        else:
            X_std = data
        # Create a PCA instance: pca
        pca1 = PCA(n_components=n_components)
        principalComponents = pca1.fit_transform(X_std)
        print('explanation variance by the selected components ', pca1.explained_variance_ratio_[:n_components].sum())
        PCA_components = pd.DataFrame(principalComponents)

        features = range(pca1.n_components_)

        PCA_components_i = pd.DataFrame(columns=PCA_components.columns)
        PCA_components_o = pd.DataFrame(columns=PCA_components.columns)
        from pyod.models import pca
        clf = pca.PCA(n_components=n_components)
        clf.fit(data)
        outliers = clf.predict(data)
        for k, i in enumerate(outliers):
            if i == 0:
                PCA_components_i.loc[k] = PCA_components.iloc[k]
            elif i == 1:
                PCA_components_o.loc[k] = PCA_components.iloc[k]

        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

        ax[0].bar(features, pca1.explained_variance_ratio_, color='black')
        ax[0].set(xlabel='variance %', ylabel='PCA features', xticks=features)

        ax[1].scatter(PCA_components_i[0], PCA_components_i[1], color='b')
        ax[1].scatter(PCA_components_o[0], PCA_components_o[1], color='r')
        for i in range(PCA_components.shape[0]):
            ax[1].text(x=PCA_components.iloc[i, 0] + 0.3, y=PCA_components.iloc[i, 1] + 0.3, s=i + 1)
        ax[1].set(xlabel='PCA 1', ylabel='PCA 2')

        plt.show()
        return (PCA_components)
    else:
        if standarize == True:
            # Standardize the data to have a mean of ~0 and a variance of 1
            X_std = StandardScaler().fit_transform(data)
        else:
            X_std = data
        # Create a PCA instance: pca
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(X_std)
        print('explanation variance by the selected components ', pca.explained_variance_ratio_[:n_components].sum())
        principalDf = pd.DataFrame(data=principalComponents[:, :2]
                                   , columns=['principal component 1', 'principal component 2'])

        features = range(pca.n_components_)

        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

        ax[0].bar(features, pca.explained_variance_ratio_, color='black')
        ax[0].set(xlabel='variance %', ylabel='PCA features', xticks=features)

        targets = np.unique(Y)
        colors = ['r', 'g']
        tar = pd.DataFrame(Y)
        finalDf = pd.concat([principalDf, tar], axis=1)

        for target, color in zip(targets, colors):
            indicesToKeep = finalDf[0] == target

            ax[1].scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                          , finalDf.loc[indicesToKeep, 'principal component 2']
                          , c=color
                          )
        for i in range(principalDf.shape[0]):
            ax[1].text(x=principalDf.loc[i, 'principal component 1'] + 0.3,
                       y=principalDf.loc[i, 'principal component 2'] + 0.3, s=i + 1)
        ax[1].set(xlabel='PCA 1', ylabel='PCA 2')

        plt.show()
        return (principalComponents)


def PLS_DA(data, Y, scale=False, n_components=10):
    """
      return the project

      :param: data: dataframe containg the independent variables
      :param Y: the dependent variable
      :param scale: if set to true independent variables will be automaticly standazized
      :return: scree plot (left fig) and xscore plot (right fig)

    """
    # Data initialisation and checks ----------------------------------------------

    assert isinstance(Y, list) or Y is None or type(Y).__module__ == np.__name__, "target data msut be a factor "
    assert isinstance(scale, bool), 'scale must be boolean instance'

    if scale == True:
        # Standardize the data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(df)
    else:
        X_std = df

    scoring = make_scorer(r2_score)
    kfold = KFold(n_splits=7)
    Q2_scores = []
    R2_scores = []
    for i in range(1, n_components + 1):
        pls_binary = PLSRegression(n_components=i)
        scores = cross_validate(pls_binary, X=data, y=Y, scoring=scoring, cv=7, n_jobs=-1, return_train_score=True, )
        R2_scores.append(np.mean(scores['train_score']))
        Q2_scores.append(np.mean(scores['test_score']))

    features = np.arange(len(Q2_scores))

    pls_binary = PLSRegression(n_components=2)
    y_binary = Y
    X_pls = pls_binary.fit_transform(df, y_binary)[0]

    labplot = [0, 1]
    unique = list(set(y_binary))
    colors = ['r', 'g']

    # R2 score
    print('R2 score :', pls_binary.score(df, Y))

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [X_pls[j, 0] for j in range(len(X_pls[:, 0])) if y_binary[j] == u]
        yi = [X_pls[j, 1] for j in range(len(X_pls[:, 1])) if y_binary[j] == u]
        ax[0].scatter(xi, yi, c=col, s=100, edgecolors='k', label=str(u))
    for i in range(X_pls.shape[0]):
        ax[0].text(X_pls[i, 0] + 0.3, X_pls[i, 1] + 0.3, s=i + 1)
    ax[0].set(xlabel='Latent Variable 1', ylabel='Latent Variable 2', title='PLS cross-decomposition')
    ax[0].legend()

    ax[1].bar(features - 0.2, R2_scores, 0.4, label='R2')
    ax[1].bar(features + 0.2, Q2_scores, 0.4, label='Q2')
    ax[1].legend()

    plt.show()
    return X_pls


