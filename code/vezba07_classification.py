#!/usr/bin/env python3
#
# Copyright 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, May 2019

"""
Digital Audio Systems

Excercise 07: Sound source classification.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from sklearn import neural_network
from sklearn import metrics
import das

# %% load data
audio_path = '../audio/'
file_names = [
        'Glas.wav',
        'Solzi.wav',
        'Violina.wav',
        'Tapan.wav',
        'Zurla.wav',
        ]
y_labels = ['voice', 'guitar', 'violin', 'drum', 'zurla']
n_labels = len(y_labels)
xs = []
for file_name in file_names:
    fs, x = wavfile.read(audio_path+file_name)
    print(fs)
    x = x / 2**15
    x = x[: min([x.size, fs*5])]
    xs.append(x)

# %% data exploration - visualization

# %% structure data
test_coef = .5  # 50%
x_trains = []
x_tests = []
for x in xs:
    x_len = x.size
    test_len = int(x_len * test_coef)
    x_train = x[:-test_len]
    x_test = x[-test_len:]
    x_trains.append(x_train)
    x_tests.append(x_test)

# %% feature extraction - spectrograms
x_train = None
x_test = None
y_train = None
y_test = None
for i, (x_train_sig, x_test_sig) in enumerate(
        zip(x_train_sigs, x_test_sigs)):
    __, __, spectrogram = das.get_spectrogram(
            fs, x_train_sig, n_win=256, plot=False)
    x_train_feats = spectrogram.T  # for sklearn
    mask_spectrogram = np.all(x_train_feats < -80, axis=1)
    x_train_feats = x_train_feats[~mask_spectrogram, :60]
    # gen y
    n_samples = x_train_feats.shape[0]
    y_train_targets = np.zeros([n_samples, n_labels])
    y_train_targets[:, i] = 1

    # same for test set
    __, __, spectrogram = das.get_spectrogram(
            fs, x_test_sig, n_win=256, plot=False)
    x_test_feats = spectrogram.T  # for sklearn
    mask_spectrogram = np.all(x_test_feats < -80, axis=1)
    x_test_feats = x_test_feats[~mask_spectrogram, :60]
    n_samples = x_test_feats.shape[0]
    y_test_targets = np.zeros([n_samples, n_labels])
    y_test_targets[:, i] = 1

    # accumulate
    if x_train is None:
        x_train = x_train_feats
        x_test = x_test_feats
        y_train = y_train_targets
        y_test = y_test_targets
    else:
        x_train = np.concatenate((x_train, x_train_feats), axis=0)
        x_test = np.concatenate((x_test, x_test_feats), axis=0)
        y_train = np.concatenate((y_train, y_train_targets), axis=0)
        y_test = np.concatenate((y_test, y_test_targets), axis=0)
        
# %% plot features
x_axis = np.arange(x_train.shape[0])
y_axis = np.arange(x_train.shape[1])
plt.figure()
plt.imshow(x_train.T,
           aspect='auto',
           origin='lower',
           extent=[0, x_axis[-1], 0, y_axis[-1]],
           vmin=-100,)
plt.plot(y_train*50, lw=2)

x_axis = np.arange(x_test.shape[0])
y_axis = np.arange(x_test.shape[1])
plt.figure()
plt.imshow(x_test.T,
           aspect='auto',
           origin='lower',
           extent=[0, x_axis[-1], 0, y_axis[-1]],
           vmin=-100,)
plt.plot(y_test*50, lw=2)

# %% shuffle data
np.random.seed(42)
train_ind_shuffle = np.random.permutation(x_train.shape[0])
x_train = x_train[train_ind_shuffle]
y_train = y_train[train_ind_shuffle]
test_ind_shuffle = np.random.permutation(x_test.shape[0])
x_test = x_test[test_ind_shuffle]
y_test = y_test[test_ind_shuffle]

# %% init and fit MLP
mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(60, 20, 10),
        activation='relu',
        learning_rate_init=0.01,
        alpha=1e-4,
        max_iter=3000,
        tol=1e-9,
        n_iter_no_change=40,
        early_stopping=False,
        shuffle=True,
        random_state=42,
        verbose=1)
mlp.fit(x_train, y_train)

# %% test accuracy
accuracy = mlp.score(x_test, y_test)
print(accuracy)


# %% calculate and plot confusion matrix
# adapted from
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
y_pred_prob = mlp.predict_proba(x_test)
cm = metrics.confusion_matrix(
        y_test.argmax(axis=1),
        y_pred_prob.argmax(axis=1))
print(cm)

# normalise
cm = cm / cm.sum(axis=0)
print(cm)

# plot heat map
fig, ax = plt.subplots()
im = ax.imshow(cm, aspect='auto', interpolation='nearest', vmax=1, vmin=0)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=y_labels,
       yticklabels=y_labels,
       ylabel='Ground truth',
       xlabel='Prediction')
fig.colorbar(im)
fig.tight_layout()

