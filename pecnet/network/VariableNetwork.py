import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from pecnet.models import *



class VariableNetwork():

    """
    A network designed to handle inputs belonged to one feature of input data and create a sequence of 
    cascaded networks based on its frequency bands.

    Variable Network, which is used in a predictive modeling pipeline to handle different frequency bands 
    of one feature from input data, keeps predictions, errors and compensated predictions generated from 
    its cascaded networks. It includes both training and testing modes, and it can be instantiated iteratively
    to handle multi-dimensional input data.

    Attributes:
        models (list): A list to store instances of the BasicNN of each cascaded network.
        mode (str): The operational mode of the network, either 'train' or 'test'.
        __predictions (list): A list of predictions from each cascaded network.
        __target_values (list): A list of error values (first is ground truth) as target values for each cascaded network.
        __compensated_predictions (list): A list of compensated predictions after each network.
        __model_index (int): Index to track the current model inside the cascaded networks.
        __is_network_initialized (bool): Flag indicating if any network has been initialized.

    Methods:
        __init__: Initializes the VariableNetwork instance and calls init_network with input and output data.
        init_network: Sets up the cascaded network structure with the provided input and output data dividing them w.r.t frequency
                      bands and statistics.
        __add_cascaded_network: Adds a cascaded network layer and performs testing or training.
        get_compensated_errors: Returns the compensated errors for the error network.
        get_last_compensated_predictions: Returns the last set of compensated predictions.
        get_predictions: Returns all predictions from the cascaded networks inside Variable Network.
        get_Last_target_values: Returns the last set of target values (errors) for next network.
        switch_mode: Switches between 'train' and 'test' modes.
    """    

    def __init__(self, X_bands,y_band):
        
        self.models = []     # List to store model instances
        self.mode = 'train'  # Default mode is train

        self.init_network(X_bands, y_band)

    def init_network(self, X_bands, y_band):

        if X_bands.ndim!=4 or y_band.ndim!=2:
            raise ValueError("x should be 4D array, y should be 2D array. Please reshape them before proceeding.")

        data_length,frequencies,statistics,sequences=X_bands.shape      
        
        self.__model_index=0                   # index of the model in the cascaded network
        self.__is_network_initialized=False    # is any cascaded network initialized yet?
        self.__predictions=[]                  # predictions at Index 0 are target value predictions, others are error predictions
        self.__target_values=[]                # labeled data and train errors of each cascaded network.Index 0 will keep actual data.Others will be errors.
        self.__compensated_predictions=[]      # compensated predictions after each cascaded network

        self.__target_values.append(y_band)

        for frequency in range(frequencies):
            for statistic in range(statistics):
                
                cascad_data_x=X_bands[:,frequency,statistic,:]
                
                if np.all(cascad_data_x == 0): # all values will be 0 for frequency=1 and statistic="std"
                    continue; 

                print("Mode: ", self.mode, " Cascaded Neural Network for frequency {} and statistic {} is working...".format(frequency,statistic))

                preds,errors,compensated_preds=self.__add_cascaded_network(cascad_data_x,self.__target_values[-1])
                
                self.__predictions.append(preds)
                self.__target_values.append(errors)
                self.__compensated_predictions.append(compensated_preds)
        
                
    def __add_cascaded_network(self, x,y): # x,y are 2D numpy arrays.
        """
        Private method which handles adding cascaded network layers into Variable Network

        This method is called iteratively for each frequency band and each sampling statistic to 
        generate cascaded neural networks inside the Variable Network. It initializes and
        trains a new model or loads an existing model for testing, based on the current mode of
        the network.It uses errors of previous network as target values for the current network.

        Args:
            x (numpy.ndarray): Input data for the network.
            y (numpy.ndarray): Target output data for the network.

        Returns:
            tuple: A tuple containing the predictions, errors, and compensated predictions from the current cascaded network.
        """
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        samp_size=x.shape[0]
        input_seq_size=x.shape[1]
        output_seq_size=y.shape[1]

        if self.mode=='train':

            model=BasicNN(sample_size=samp_size,input_sequence_size=input_seq_size,output_sequence_size=output_seq_size)            
            model.fit(x,y)
            self.models.append(model)  # Store the trained model
        
        elif self.mode=='test':

            model = self.models[self.__model_index]  # Get the model from the list
            self.__model_index += 1                  # Increment the model index   
        
        cascad_pred=model.predict(x)    

        if not self.__is_network_initialized:
            
            cascad_comp_pred=cascad_pred
            cascad_err=cascad_pred-y
            self.__is_network_initialized=True

            return cascad_pred,cascad_err,cascad_comp_pred      
    
        cascad_comp_pred=self.__compensated_predictions[-1]-cascad_pred
        cascad_err=cascad_pred-self.__target_values[-1]


        return cascad_pred,cascad_err,cascad_comp_pred

    def get_compensated_errors(self):
        """
        Returns the compensated errors for the error network. ----> compensated_train-y_train
        """

        return self.__compensated_predictions[-1]-self.__target_values[0]     

    def get_last_compensated_predictions(self):
        """
        Returns the last compensated predictions for the next network.
        """
        return self.__compensated_predictions[-1]                             
    
    def get_predictions(self):
        """
        Returns predictions of all cascaded networks inside Variable Network for final network
        """
        return self.__predictions                                             

    def get_Last_target_values(self):
        """
        Returns last error target values (errors of errors) for next variable network with escaping error network.
        """
        return self.__target_values[-1]                                       
    
    def switch_mode(self,mode):
        self.mode=mode