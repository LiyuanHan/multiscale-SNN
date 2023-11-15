##
import numpy as np

from tqdm.auto import trange

from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

##
filepath = '/data/data_hly/Multiscale_SNN_Code/'


##
def load_data(bin_size, expect_num_bins):
	# R = data['R'][0]  # the infamous Shenoy lab "R" struct
	
	# Extract the targets
	targets = np.row_stack([Ri['target'].ravel() for Ri in R])
	unique_targets, conditions = np.unique(targets, return_inverse=True, axis=0)
	
	# Extract the spikes and positions in 10ms bins
	# def bin_spikes(trial):
	# 	spikes = R[trial]['spikeRaster'].toarray()
	# 	num_neurons, num_frames = spikes.shape
	# 	num_bins = int(np.ceil(num_frames / bin_size[trial]))
	#
	# 	pad = np.zeros((num_neurons, num_bins * bin_size[trial] - num_frames))
	# 	padded_spikes = np.column_stack([spikes, pad])
	# 	if padded_spikes.shape[1] < bin_size[trial]*expect_num_bins:
	# 		pad2 = np.zeros((num_neurons, bin_size[trial]*expect_num_bins - padded_spikes.shape[1]))
	# 		padded_spikes = np.column_stack([padded_spikes, pad2])
	# 	if int(padded_spikes.shape[1] / bin_size[trial]) == expect_num_bins - 1:
	# 		pad2 = np.zeros((num_neurons, bin_size[trial]))
	# 		padded_spikes = np.column_stack([padded_spikes, pad2])
	#
	# 	# print('trial:',trial, 'num_bins: ',num_bins,'bin_size: ',bin_size[trial],'padded_spikes_shape: ',padded_spikes.shape)
	# 	return padded_spikes[:,:expect_num_bins*bin_size[trial]].reshape(num_neurons, expect_num_bins, bin_size[trial]).sum(-1)
	
	def bin_spikes(trial):
		spikes = R[trial]['spikeRaster'].toarray()
		num_neurons, num_frames = spikes.shape
		
		if num_frames >= expect_num_bins:
			return spikes[:,:expect_num_bins]
		
		pad = np.zeros((num_neurons, expect_num_bins * bin_size[trial] - num_frames))
		padded_spikes = np.column_stack([spikes, pad])
		
		if int(padded_spikes.shape[1] / bin_size[trial]) == expect_num_bins - 1:
			pad2 = np.zeros((num_neurons, bin_size[trial]))
			padded_spikes = np.column_stack([padded_spikes, pad2])
		
		# print('trial:',trial, 'num_bins: ',num_bins,'bin_size: ',bin_size[trial],'padded_spikes_shape: ',padded_spikes.shape)
		return padded_spikes[:, :expect_num_bins * bin_size[trial]].reshape(num_neurons, expect_num_bins,
		                                                                    bin_size[trial]).sum(-1)
	
	def bin_positions(trial, smooth=20):
		pos = R[trial]['cursorPos'][:2]
		dim, num_frames = pos.shape
		num_bins = int(np.ceil(num_frames / bin_size[trial]))
		if num_bins < expect_num_bins:
			num_bins = expect_num_bins
		# if num_bins == expect_num_bins - 1:
		# 	num_bins += 1
		smooth_pos = gaussian_filter1d(pos, smooth, axis=1)
		f = interp1d(np.arange(num_frames), smooth_pos, axis=-1, fill_value="extrapolate")
		pos = f(np.arange(num_bins) * bin_size[trial])
		return pos[:,:expect_num_bins]
	
	# Get the spikes, cursor positions, and cursor velocities for each trial
	all_data = []
	for trial in trange(len(R)):
		spk = bin_spikes(trial).T
		pos = bin_positions(trial).T
		
		vel = np.gradient(pos, axis=0)
		con = conditions[trial]
		# if not (con == 4):
		print(f"spikes.shape={spk.shape}, pos.shape={pos[:expect_num_bins,:].shape}")
		all_data.append(dict(spikes=spk,
		                     position=pos[:expect_num_bins,:],
		                     velocity=vel,
		                     condition=con))
	
	return all_data


##
dataFile = filepath + "Data/JR_2015-12-04_truncated2.mat"
data = loadmat(dataFile)
expect_num_bins = 40
bin_size = []
R = data['R'][0]  # the infamous Shenoy lab "R" struct

for trial in trange(len(R)):
	spikes = R[trial]['spikeRaster'].toarray()
	num_neurons, num_frames = spikes.shape
	bin_size.append(int(25))
# print(bin_size)

all_data = load_data(bin_size, expect_num_bins)

## divided into 0，1，..., 8 trials
all_trial_index = [[], [], [], [], [], [], [], [], []]
for i in range(len(all_data)):
	all_trial_index[all_data[i]['condition']].append(i)


##
index_re_sort = [0, 2, 5, 7, 8, 6, 3, 1]
all_data_spike_sort = np.zeros((all_data[all_trial_index[0][0]]['spikes'].T.shape[0], 1)) + 999
all_data_pos_sort = np.zeros((all_data[all_trial_index[0][0]]['position'].T.shape[0], 1)) + 999
for k in range(30):
	for i in index_re_sort:
		spk = all_data[all_trial_index[i][k]]['spikes'].T
		pos = all_data[all_trial_index[i][k]]['position'].T
		all_data_spike_sort = np.column_stack((all_data_spike_sort, spk))
		all_data_pos_sort = np.column_stack((all_data_pos_sort, pos))
		
		spk_2 = all_data[all_trial_index[i][k] + 1]['spikes'].T
		pos_2 = all_data[all_trial_index[i][k] + 1]['position'].T
		all_data_spike_sort = np.column_stack((all_data_spike_sort, spk_2))
		all_data_pos_sort = np.column_stack((all_data_pos_sort, pos_2))

## save .mat files
import scipy.io as scio

data_for_BCI_RLS = {'spike': all_data_spike_sort[:, 1:],
                    'position': all_data_pos_sort[:, 1:]
                    }
scio.savemat(filepath + 'Data/data_for_BCI_RLS-v2.mat', data_for_BCI_RLS)
print('saved finished!')
##


