# multiscale-SNN
## Datasets
### 1. task-sine_popularity-v2.mat
- This is the dataset for point-to-point control task.

### 2. lor_data_4e-5.mat
- This is the dataset for Lorenz system task. It can be founed and downloaded from the Tag.

### 3. JR_2015-12-04_truncated2.mat
- The publicly accessible dataset [1] consists of a 96-channel Utah array recording from premotor cortex of a rhesus monkey performing a center-out-and-back task.
- [1] V. Gilja, P. Nuyujukian, C. A. Chestek, J. P. Cunningham, B. M. Yu, J. M. Fan, M. M. Churchland, M. T. Kaufman, J. C. Kao, S. I. Ryu et. al., A high-performance neural prosthesis enabled by control algorithm design, Nature neuroscience, vol. 15, no. 12, pp. 1752--1757, 2012.

### 4. data_for_BCI_RLS.mat
- This dataset is the result of processing a publicly available dataset (JR_2015-12-04_truncated2.mat).

## Codes
### 1. x0_xT_DP_True.m
- This file is the function of DDP algorithm.

### 2. IZ_Task1_sine_popularity.m
- This code is for point-to-point control task using Izhikevich model.

### 3. IZ_Task1_Lorenz.m
- This code is for Lorenz system task using Izhikevich model.

### 4. LIF_Task1.m
- This code is for point-to-point control task using LIF model.

### 5. LIF_Task1_Lorenz.m
- This code is for Lorenz system task using LIF model.

### 6. data_preprocess_for_RLS.py
- This python file is for processing the publicly available dataset (JR_2015-12-04_truncated2.mat).

### 7. IZ_Task_BCI.m
- This code is for the center-out-and-back task in an actual BCI scenario using exclusively Izhikevich neurons in our SNN model.

### 8. IZ_Task1_BCI_pure.m
- This code is for the center-out-and-back task in an actual BCI scenario using exclusively biological neurons recorded by the BCI in our SNN model.

### 9. IZ_Task1_BCI_mixed.m
- This code is for the center-out-and-back task in an actual BCI scenario using mixed neurons recorded in our SNN model. These mixed neurons comprise both biological neurons and Izhikevich neurons, with the latter simulating neurons that are not directly recorded by the BCI electrode.

### 10. plot_R2_value.m
- This code is plotting for R2 scores computed by three scenarios, i.e., Izhi. Bio. Mixded.

### 
