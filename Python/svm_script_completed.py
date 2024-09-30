#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#Q1 et Q2

#%%
###############################################################################
#               Iris Dataset
###############################################################################



# Charger les données Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Prétraitement : Standardisation et sélection des classes
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[y != 0, :2]
y = y[y != 0]

# Nombre d'itérations
n_iterations = 50

# Variables pour stocker les scores
scores_linear_train = []
scores_linear_test = []

# Boucle pour exécuter l'évaluation 50 fois
for i in range(n_iterations):
    # Mélange et split des données
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)

    # on change "linear" en "poly" pour la Q2
    clf_linear = SVC(kernel='poly', C=1.0)  

    clf_linear.fit(X_train, y_train)

    # Calcul des scores
    train_score = clf_linear.score(X_train, y_train)
    test_score = clf_linear.score(X_test, y_test)

    # Stockage des scores
    scores_linear_train.append(train_score)
    scores_linear_test.append(test_score)

# Calcul des moyennes des scores
mean_linear_train = np.mean(scores_linear_train)
mean_linear_test = np.mean(scores_linear_test)

# Affichage des résultats
print(f"Score moyen pour le noyau linéaire - Entraînement: {mean_linear_train}")
print(f"Score moyen pour le noyau linéaire - Test: {mean_linear_test}")

# PARTIE 2 LES VISAGES


#%%
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
                              # data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#| echo = FALSE
#| eval = TRUE

# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

#| echo = FALSE
#| eval = TRUE
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()
#%%
# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_train, y_train)) 

ind = np.argmax(scores)
best_C = Cs[ind]
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()
#%%
# predict labels for the X_test images with the best classifier

# fait par moi 
t0=time()
clf= SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train,y_train)

# fin du fait par moi, vérifier avec les autres

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib
# ici on a un soucis de taille de vecteurs donc on est obligés de les reshape
y_pred = clf.predict(X_test)
prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#%%
# Q5

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Fonction pour exécuter le SVM avec une validation croisée et retourner les scores
def run_svm_cv(_X, _y):
    # Séparation aléatoire en ensembles d'entraînement et de test
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    # Paramètres à tester pour le modèle SVM
    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}  # de 0.001 à 1000
    _svr = svm.SVC()  # Modèle SVM avec noyau linéaire
    _clf_linear = GridSearchCV(_svr, _parameters)  # Recherche du meilleur C
    _clf_linear.fit(_X_train, _y_train)  # Entraînement

    # Retourne les scores d'entraînement et de test
    train_score = _clf_linear.score(_X_train, _y_train)
    test_score = _clf_linear.score(_X_test, _y_test)
    return train_score, test_score

# Nombre d'itérations
n_iterations = 50

# Variables pour stocker les scores
scores_sans_bruit_train = []
scores_sans_bruit_test = []
scores_avec_bruit_train = []
scores_avec_bruit_test = []

# Nombre d'échantillons (ajusté en fonction des données)
n_samples = X.shape[0]

# Boucle pour exécuter l'évaluation 100 fois
for i in range(n_iterations):
    # Exécution sans variables de nuisance
    train_score, test_score = run_svm_cv(X, y)
    scores_sans_bruit_train.append(train_score)
    scores_sans_bruit_test.append(test_score)

    # Ajout de variables de nuisance (bruit)
    n_features = X.shape[1]
    sigma = 1  # Écart-type du bruit
    noise = sigma * np.random.randn(n_samples, 300)  # Génération du bruit gaussien avec 300 nouvelles variables
    X_noisy = np.concatenate((X, noise), axis=1)  # Ajout des variables bruitées aux données originales

    # Mélange des données bruitées pour éviter tout biais
    X_noisy = X_noisy[np.random.permutation(X_noisy.shape[0])]

    # Exécution avec les variables de nuisance
    train_score, test_score = run_svm_cv(X_noisy, y)
    scores_avec_bruit_train.append(train_score)
    scores_avec_bruit_test.append(test_score)

# Calcul des moyennes des scores
mean_score_sans_bruit_train = np.mean(scores_sans_bruit_train)
mean_score_sans_bruit_test = np.mean(scores_sans_bruit_test)
mean_score_avec_bruit_train = np.mean(scores_avec_bruit_train)
mean_score_avec_bruit_test = np.mean(scores_avec_bruit_test)

# Affichage des résultats
print(f"Score moyen sans variables de nuisance - Entraînement: {mean_score_sans_bruit_train}")
print(f"Score moyen sans variables de nuisance - Test: {mean_score_sans_bruit_test}")
print(f"Score moyen avec variables de nuisance - Entraînement: {mean_score_avec_bruit_train}")
print(f"Score moyen avec variables de nuisance - Test: {mean_score_avec_bruit_test}")



#%%
# Q6
print("Score après réduction de dimension avec PCA")

n_components = 20  # Nombre de composantes principales, à ajuster selon la performance
pca = PCA(n_components=n_components, svd_solver='randomized').fit(X_noisy)

# Transformation des données avec les composantes principales
X_pca = pca.transform(X_noisy)

# Affichage de la variance expliquée par chaque composante 
print(f"Variance expliquée par les {n_components} composantes principales : {np.sum(pca.explained_variance_ratio_):.2f}")

# Exécution de run_svm_cv avec les données transformées par la PCA
run_svm_cv(X_pca, y)
# %%
