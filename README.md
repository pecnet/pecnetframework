<h1 align="center">PREDICTIVE ERROR COMPENSATED WAVELET NEURAL NETWORKS (PECNET)</h1>

**PECNET** is an innovative artificial neural network arhitecture that addresses the issue of overfitting by training the data separately in cascaded networks based on different frequency bands and types, utilizing the remaining errors of each network instead of utilizing outputs as the target label. In the PECNET architecture, data is fed into the networks with a low-frequency band in a wide time window, and the subsequent network is trained with narrower time windows and high-frequency data while utilizing the error from the previous low-frequency network. This approach enhances the orthogonality of data characteristics across time windows and improves prediction performance as additional networks are added, reducing the likelihood of overfitting. Additionally, by applying wavelet transform to the various frequency components of each network, it is possible to distinguish and extract the variety of patterns present in the data with the analysis at different levels of time and frequency resolution. These procedures can be followed in Figure 1.



![Figure 1](https://github.com/user-attachments/assets/373d9352-fb7d-4641-b709-efb461cbef9e)
<p align="center"><em>Figure 1: PECNET Architecture</em></p>

## Core Concepts

### 🔴 Cascaded NNs with Residual Errors
### 🔵 Using Different Frequency Bands Hierarchically 
### 🟢 Wavelet Transformation as a Feature Selection Mechanism
### 🟡 Adaptive Normalization for Nonstationary Cases
### 🟤 Data Fusion in an Orthonogal Manner

<h1 align="center">SIMPLE USAGE SCENARIO</h1>
The following diagram outlines the solution steps to be followed from start to finish for a time series forecasting task, along with the corresponding code equivalents in the framework. Assuming that the framework installation is successfully completed and the data is correctly transferred to the software environment, this basic code flow demonstrates how to implement the PECNET model for many time series forecasting tasks.
<br>
<br>
In the outlined flow, univariate Apple stock data (aapl_prices) is used. This flow needs to be modified for two different scenarios:
<br>
<br>
1. When the target data is different from the input data, the target data should also be processed with the preprocess() function and then provided to the variable network in step 3.
<br>
<br>
2. For a multivariate time series, training should initially be completed using a univariate network with the feature of the input data that has the highest correlation with the target data. Then, during training, the errors generated should be accessed using the <em>get_last_variable_errors()</em>  function under the Pecnet class, and a second variable network should be added using the second feature of the data that has the highest correlation with these errors. The target data for this network should be the aforementioned errors. This procedure should be repeated until no more correlation is found or until prediction performance deteriorates.
<br>
<br>
<br>

![pecnet_flow](https://github.com/pecnet/pecnetframework/assets/156237148/fa0106dd-a7d1-4cac-b6ec-45f74b141068)
