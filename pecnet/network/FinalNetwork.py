from pecnet.models import *
from pecnet.preprocessing import DataPreprocessor

class FinalNetwork:
    """
        A network designed to integrate and finalize predictions from a cascaded sequence of models.

        This class represents a Final Network, which is used in the last stage of a predictive modeling pipeline.
        Its main function is to take all cascaded predictions and generate a final, unified prediction. The class
        supports both training and testing modes.

        Attributes:
            model (BasicNN): stores the pre-trained neural network model.
            mode (str): The operational mode of the network, either 'train' or 'test'.
            __final_predictions (numpy.ndarray): Stores the final unified predictions which is denormalized and unscaled after training or testing.

        Methods:
            __init__: Initializes the FinalNetwork instance and calls init_network with all cascaded predictions.
            init_network: Sets up the network with the provided cascaded predictions, then gets adjusted final ground truths and calls __add_final_network.
            __add_final_network: Adds a final network layer and performs testing or training.
            final_predictions: Returns the final unified predictions which is denormalized and unscaled.
            __generate_final_preds: Denormalizes and unscales the predictions.
            switch_mode: Switches between 'train' and 'test' modes.
    """    
    def __init__(self, all_cascaded_predictions):
        
        self.model = None    # keeps the model instance
        self.mode = 'train'  # Default mode is train

        self.init_network(all_cascaded_predictions)    

    def init_network(self, all_cascaded_predictions):
        
        if all_cascaded_predictions.ndim!=2 :
            raise ValueError("Prediction values should be 2D array. Please reshape before proceeding.")
        
        
        preds=self.__add_final_network(all_cascaded_predictions,DataPreprocessor().get_final_y_processed())
        self.__final_predictions=self.__generate_final_preds(preds)
             
    def __add_final_network(self, x,y): # x,y are 2D numpy arrays.  
        """
        Adds a final network layer.

        This method handles the final stage of testing or training. It initializes and trains
        a new model or loads an existing model for prediction, based on the current mode of the network.

        Args:
            x (numpy.ndarray): Input data for the network, which includes all cascaded predictions.
            y (numpy.ndarray): Target output data for the network, which is scaled and normalized ground truths.

        Returns:
            numpy.ndarray: An array containing the predictions from the final network.
        """
        samp_size=x.shape[0]
        input_seq_size=x.shape[1]
        output_seq_size=y.shape[1]
        
        print("Mode: ",self.mode, " Final Network is working...")

        if self.mode=='train':

            model=BasicNN(sample_size=samp_size,input_sequence_size=input_seq_size,output_sequence_size=output_seq_size)            
            model.fit(x,y)
            self.model=model   # Store the trained model

        elif self.mode=='test':
            model = self.model       # Get the trained model 

        preds=model.predict(x)
    
        return preds

    def final_predictions(self):
        return self.__final_predictions

    def __generate_final_preds(self,preds):
        """
        Denormalizes and unscales the predictions.
        """
        denormalization_term= DataPreprocessor().get_final_denormalization_term()

        if self.mode=='test':
            denormalization_term[-1]=DataPreprocessor().generate_final_normalization_term_for_last_pred_element()
            
        denormalized_preds=preds+denormalization_term

        if DataPreprocessor().target_normalizer:
            denormalized_preds=DataPreprocessor().target_normalizer.inverse_transform(denormalized_preds)
        
        if DataPreprocessor().target_scaler is not None:
            unscaled_preds=DataPreprocessor().target_scaler.unscale1D(denormalized_preds)
            return unscaled_preds
        else:
            return denormalized_preds



    def switch_mode(self,mode):
        self.mode=mode             
