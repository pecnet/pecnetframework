from pecnet.models import *
from pecnet.preprocessing import DataPreprocessor
from pecnet.network.ModelLoader import train_or_load_model



class ErrorNetwork:
    """
    This class represents the Pecnet Model's Error Network, which is part of a larger predictive model. Its main function
    is to predict errors in a previous prediction layer, and then provide corrected predictions to the Final Network.

    Attributes:
        models (list): A list to store instances of the predictive models for test phase.
        mode (str): The operational mode of the network, either 'train' or 'test'.
        __error_predictions (numpy.ndarray): Stores the error predictions of the error network.
        __compensated_error_predictions (numpy.ndarray): Stores the compensated error predictions.
        __target_values (numpy.ndarray): Stores the errors of errors

    Methods:
        __init__: Initializes the ErrorNetwork instance.
        init_network: Initializes and configures the network based on provided data from Variable Network.
        __add_error_network: Adds an error correction network and performs the necessary train or test operations.
        get_error_predictions: Returns the error predictions of the error network.
        get_compensated_error_predictions: Returns the compensated error predictions.
        get_target_values: Returns the errors of errors.
        switch_mode: Switches between 'train' and 'test' modes.
    """
    

    def __init__(self, error_band,last_compensated_preds):

        self.models = []     # List to store model instances
        self.mode = 'train'  # Default mode is train

        self.init_network(error_band, last_compensated_preds)

    def init_network(self, error_band,last_compensated_preds):
        
        if error_band.ndim!=2:
            raise ValueError("Error data should be 2D array. Please reshape it before proceeding.")
        
        self.__model_index=0                   # index of the model in the cascaded network

        error_X,error_Y,denormalization_term=DataPreprocessor().preprocess_errors(error_band) 

        #shifting last_compensated predictions 1 step back in time and trimming first window 
        last_compensated_preds=last_compensated_preds[DataPreprocessor().get_error_sequence_size()+1:]

        print("Mode: ", self.mode, " Error Network is working...")

        # error predictions of error network, errors of errors, compensated error predictions  
        self.__error_predictions,self.__target_values,self.__compensated_error_predictions=self.__add_error_network(error_X, error_Y,denormalization_term,last_compensated_preds)

                
    def __add_error_network(self, x,y,error_denormalizer,compensated_preds): # x,y are 2D numpy arrays.  
        """
        Adds error correction network to the PECNET model and performs the necessary train or test operation.

        Args:
            x (numpy.ndarray): The sequential error input data for the error network.
            y (numpy.ndarray): The error output data for the error network.
            error_denormalizer (numpy.ndarray): The denormalization term for the error predictions.
            compensated_preds (numpy.ndarray): The compensated predictions from the previous layer.Theoretically, it will be sended to the next cascaded layer if added.
        
        Steps:
            1. Data Preprocessing: The x and y is preprocessed to be in the correct format for
                                   model training or prediction before adding an error network. 
                                   x is formed as a sequence of errors w.r.t error sequence size 
                                   and y is formed as error outputs with time shifting
            2. Model Initialization and Training/Loading: Depending on the mode, a new model is either
                                                          trained on the preprocessed data or an existing
                                                          model is loaded from the list of models.
                                                          (Theoretically, there can be multiple error networks in the cascaded network)
            3. Error Correction Computation: Calculates the error predictions, errors of errors and compensated error predictions.
              
        """

        model, err_preds = train_or_load_model(x, y, self.mode, self.models, self.__model_index)

        if self.mode == 'test':
            self.__model_index += 1

        err_comp_preds=compensated_preds-err_preds
        err_error=err_preds-y

        return err_preds,err_error,err_comp_preds

    def get_error_predictions(self):                    #returns error predictions of error network.
        return self.__error_predictions                 
    def get_compensated_error_predictions(self):        #returns compensated error predictions of error network.
        return self.__compensated_error_predictions     
    def get_target_values(self):                        #returns errors of errors.  
        return self.__target_values 
    def switch_mode(self,mode):
        self.mode=mode                         