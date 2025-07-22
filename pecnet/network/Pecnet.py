from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from pecnet.preprocessing import DataPreprocessor

class Pecnet:
    def __init__(self):

        self.mode="train"
        self.final_network = None
        self.error_network = None
        self.variable_networks = []

    def add_variable_network(self, variable_network):
        """
        Adds a VariableNetwork instance to the Pecnet model.

        Args:
            variable_network (VariableNetwork): The variable network to be added.
        """
        self.variable_networks.append(variable_network)

    def get_target_values_for_current_variable_network(self,varnet_index=None):
        """
        Retrieves the last target values from the most recently added variable network.
        This is used as the target values for the next variable network when y_train is not provided.

        Args:
            varnet_index (int, optional): Index of the current VariableNetwork during test phase
            to retrieve correct target values hierarchically .
        Returns:
            np.ndarray: Last target values from the most recent variable network.
        """
        if not self.variable_networks:
            raise ValueError("No variable networks found. Add at least one variable network before calling this method.")

        if self.mode == "train":
            return self.variable_networks[-1].get_Last_target_values()
        elif self.mode == "test":
            if varnet_index is None or varnet_index == 0:
                raise ValueError("Variable Network index must be set correctly and must be > 0.")
            return self.variable_networks[varnet_index-1].get_Last_target_values()
        else:
            raise ValueError("Invalid mode. Choose from 'train' or 'test'.")

    def get_comp_preds_for_current_variable_network(self, varnet_index=None):
        """
        Retrieves the last compensated predictions from the previous VariableNetwork.
        Args:
            varnet_index (int, optional): Index of the current VariableNetwork to process compensated predictions.
        Returns:
            np.ndarray: Previous compensated predictions to retrieve for current VariableNetwork .
        """
        if not self.variable_networks:
            raise ValueError(
                "No variable networks found. Add at least one variable network before calling this method.")

        if self.mode == "train":
            return self.variable_networks[-1].get_last_compensated_predictions()
        elif self.mode == "test":
            if varnet_index is None or varnet_index == 0:
                raise ValueError("Variable Network index must be set correctly and must be > 0.")
            return self.variable_networks[varnet_index-1].get_last_compensated_predictions()
        else:
            raise ValueError("Invalid mode. Choose from 'train' or 'test'.")

    def get_shifted_compensated_errors(self):
        """
        Adjusts timestamps of compensated errors to be used in error network.
        
        Returns: 
            List[float] : Errors shifted 1 step back in time.
        """
        return self.variable_networks[-1].get_compensated_errors()[:-1]
    
    def get_last_compensated_predictions(self):
        """
        Just a Wrapper method, feeds last compensated predictions to the next network.
        Returns: 
            List[float] : Last compensated predictions in pipeline.
        """
        return self.variable_networks[-1].get_last_compensated_predictions()
    
    def get_all_preds(self):
        """
        Just a Wrapper method, feeds all predictions included error predictons to the final network.Adjusts sizes.
        Returns: 
            List[List] : All predictions in pipeline.
        """
        #cascade predictions will be trimmed for final network,1 for time shifting, others for error sequence size 
        adjust_size=DataPreprocessor().get_error_sequence_size()+1
        adjusted_preds = []

        for varnet in self.variable_networks:
            preds = [p[adjust_size:] for p in varnet.get_predictions()]
            adjusted_preds.extend(preds)
        
        #add error predictions to the end of the prediction list
        adjusted_preds.append(self.error_network.get_error_predictions())

        X_final=np.concatenate(adjusted_preds,axis=1)
        
        return  X_final

    def predict(self,
                *test_inputs: np.array,
                test_target: np.array) -> np.array:

        """Tries to predict test_target using test_input values to generate errors for error network,they comes 1 step back in time.
        Args:  
            test_inputs: The test data to be used for prediction.
            test_target: Ground truth  values to be used for error calculation.
        Returns:
            np.ndarray: final predictions of the pecnet model.
        """

        #switches mode to test, so that it can be used for prediction
        self.switch_mode("test")

        #predicts test_target using hierarchical networks fed with multivariate data
        for i, varnet in enumerate(self.variable_networks):

            if i == 0:
                varnet.init_network(test_inputs[i], test_target)
            else:
                y_target = self.get_target_values_for_current_variable_network(i)
                pre_comp = self.get_comp_preds_for_current_variable_network(i)
                varnet.init_network(test_inputs[i], y_target, pre_comp)

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
        # calculate mae
        mae = mean_absolute_error(y, pred)
        
        return f"RMSE: {round(rmse,3)} R2 : {round(r2,3)},MAE: {round(mae,3)}, MAPE: {round(mape,3)}"

    def switch_mode(self, mode):
        """
        Switches mode of the network.
        Args:
            mode: "train" or "test"
        """
       
        for varnet in self.variable_networks:
            varnet.switch_mode(mode)
        self.error_network.switch_mode(mode)
        self.final_network.switch_mode(mode)

        DataPreprocessor().switch_mode(mode)

        self.mode= mode;