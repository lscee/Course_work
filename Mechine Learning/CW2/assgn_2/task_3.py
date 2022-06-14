import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *
#from task_1 import X_phoneme_1

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for data,i in zip(f1,range(len(f1))): 
    X_full[i][0] = data
if(len(f2) != len(f1)):
    print("data erro")
for data,i in zip(f2,range(len(f2))): 
    X_full[i][1] = data

########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
p_id=[1,2]
# X_phonemes_1_2 = ...

X_phonemes_1_2 = np.zeros((np.sum(phoneme_id == p_id[0])+(np.sum(phoneme_id == p_id[1])), 2))
#print(len(X_phoneme))

phoneme_1_2 = []
index=0
for id,i in zip(phoneme_id,range(len(X_full))):
    if id in p_id:
        X_phonemes_1_2[index] = X_full[i]
        phoneme_1_2.append(id)
        index += 1
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
########################################/

# Plot array containing the chosen phoneme

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme {}'.format(p_id)
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 or 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phoneme_{}.png'.format(p_id))
plt.savefig(plot_filename)

# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

pretrain_file = ['data/GMM_params_phoneme_01_k_03.npy', 'data/GMM_params_phoneme_02_k_03.npy', 'data/GMM_params_phoneme_01_k_06.npy',  'data/GMM_params_phoneme_02_k_06.npy']

GMM_parameters1 = np.load(pretrain_file[2], allow_pickle=True)
GMM_parameters1 = np.ndarray.tolist(GMM_parameters1)
GMM_parameters2 = np.load(pretrain_file[3], allow_pickle=True)
GMM_parameters2 = np.ndarray.tolist(GMM_parameters2)


mu1 = GMM_parameters1['mu']
s1 = GMM_parameters1['s']
p1 = GMM_parameters1['p']

mu2 = GMM_parameters2['mu']
s2 = GMM_parameters2['s']
p2 = GMM_parameters2['p']

# as dataset X, we will use only the samples of the chosen phoneme
X = X_phonemes_1_2.copy()
# get number of samples
N = X.shape[0]
# get dimensionality of our dataset
D = X.shape[1]

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z = np.zeros((N,k)) # shape Nxk

# Do the E-step
Z1 = get_predictions(mu1, s1, p1, X)
Z2 = get_predictions(mu2, s2, p2, X)



ax1.clear()
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 or 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# Plot gaussians after each iteration


accuracy=0
########################################/
err1 = 0
err2 = 0
accuracy1 = 0
accuracy2 = 0
for i in range(len(X)):
    pred = 1 if max(Z1[i])> max(Z2[i]) else 2
    if pred != 1 and phoneme_1_2[i] ==1:
        err1+=1
    if pred !=2 and phoneme_1_2[i] ==2:
        err2+=1

accuracy1 = 1 - err1/np.sum(phoneme_id==p_id[0])
accuracy2 = 1 - err2/np.sum(phoneme_id==p_id[1])
accuracy = 1 - (err1 + err2)/ (np.sum(phoneme_id==p_id[0])+ np.sum(phoneme_id==p_id[1]))
print("phoneme 1 accuracy:{:.2f}%".format(accuracy1*100))
print("phoneme 2 accuracy:{:.2f}%".format(accuracy2*100))
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy*100))
#print(mu)
plot_gaussians(ax1, 2*s1, mu1)
plot_gaussians(ax1, 2*s2, mu2)
################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()