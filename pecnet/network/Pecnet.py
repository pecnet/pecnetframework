from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from pecnet.preprocessing import DataPreprocessor

class Pecnet:
    def __init__(self):
        self.final_network = None
        self.error_network = None
        self.variable_network = None
    
    def get_shifted_compensated_errors(self):
        """
        Adjusts timestamps of compensated errors to be used in error network.
        
        Returns: 
            List[float] : Errors shifted 1 step back in time.
        """
        return self.variable_network.get_compensated_errors()[:-1]
    
    def get_last_compensated_predictions(self):
        """
        Just a Wrapper method, feeds last compensated predictions to the next network.
        Returns: 
            List[float] : Last compensated predictions in pipeline.
        """
        return self.variable_network.get_last_compensated_predictions()
    
    def get_all_preds(self):
        """
        Just a Wrapper method, feeds all predictions included error predictons to the final network.Adjusts sizes.
        Returns: 
            List[List] : All predictions in pipeline.
        """
        #cascade predictions will be trimmed for final network,1 for time shifting, others for error sequence size 
        adjust_size=DataPreprocessor().get_error_sequence_size()+1
        adjusted_preds= [preds[adjust_size:] for preds in self.variable_network.get_predictions()]
        
        #add error predictions to the end of the prediction list
        adjusted_preds.append(self.error_network.get_error_predictions())

        X_final=np.concatenate(adjusted_preds,axis=1)
        
        return  X_final

    def predict(self,
                X_test: np.array,
                test_truth: np.array) -> np.array:

        """Predicts X_test and uses test_truth values to generate errors for error network,they comes 1 step back in time.
        Args:  
            X_test: The test data to be predicted.
            test_truth: Ground truth  values to be used for error calculation.
        Returns:
            List of np.arrays: [prediction, time_shifted_test_truth]
        """

        #switches mode to test, so that it can be used for prediction
        self.switch_mode("test")

        #predicts X_test TODO: add multiview support, that'll also solve network interaction problems easily

        self.variable_network.init_network(X_test, test_truth)
        self.error_network.init_network(self.get_shifted_compensated_errors(),self.get_last_compensated_predictions())
        self.final_network.init_network(self.get_all_preds())


        return self.final_network.final_predictions()

    # Evaluate the model    
    def evaluate(self, pred, y):
        """
        This function takes the predicted values and the actual values, aligns them for comparison,\
        and computes several common metrics to assess the accuracy and performance of the model.\
        It calculates the Root Mean Squared Error (RMSE), R-squared (R2) score, and the Mean Absolute\
        Percentage Error (MAPE).
            
        Note:
            The function adjusts the length of the actual values array to match the length of the
            predictions array due to the pre-required timestamps size for prediction process and time shifting of the error sequences.        
        """
        pred=pred[:-1] # last value is the tomorrow's value, it's not used in the evaluation
        
        #Adjust real prices index to match with predictions
        if len(y)>len(pred):
            y=y[-len(pred):]
            
        # calculate rmse
        rmse = np.sqrt(mean_squared_error(y, pred))
        # calculate r2
        r2 = r2_score(y, pred)
        # calculate mape
        mape = mean_absolute_percentage_error(y, pred)
        
        return f"RMSE: {round(rmse,3)} R2 : {round(r2,3)}, MAPE: {round(mape,3)}"

    def switch_mode(self, mode):
        """
        Switches mode of the network.
        Args:
            mode: "train" or "test"
        """
       
        self.variable_network.switch_mode(mode)
        self.error_network.switch_mode(mode)
        self.final_network.switch_mode(mode)

        DataPreprocessor().switch_mode(mode)