from typing import List
from pywt import wavedec
import pandas as pd
from pecnet.preprocessing.Normalizers import *

class DataPreprocessor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataPreprocessor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'): # Check if instance is already initialized
            print("Initializing DataPreprocessor")
            self.initialized = True
            self.mode="train"
            self.scaler=Scaler()
            self.window_normalizer=WindowNormalizer()
            self.normalizer=Normalizer()

    def preprocess(self,
                   data: np.ndarray, 
                   sampling_periods:List[int] = [1,4], 
                   sampling_statistics:List[str] = ["mean"], 
                   sequence_size:int = 4, 
                   error_sequence_size:int = 4, 
                   wavelet_type:str ="haar",
                   test_ratio:float = 0.2
                   ) -> List[np.ndarray]:
        """
            Wrapper method --> Preprocesses the given data using wavelet transform and statistical analysis.

            Args:
                data (np.ndarray): The data to be processed, as a NumPy array.
                sampling_periods (List[int], optional): List of sampling periods. Defaults to [1, 4].
                sampling_statistics (List[str], optional): List of sampling statistics to be calculated, like 'mean'. Defaults to ["mean"].
                sequence_size (int, optional): The size of the sequence for processing. Defaults to 4.
                error_sequence_size (int, optional): The size of the sequence for error network. Defaults to 4.
                wavelet_type (str, optional): Type of the wavelet to be used. Defaults to "haar".
                test_ratio (float, optional): The ratio of the test data. Defaults to 0.2.

            Returns:
                List[np.ndarray]: A list of NumPy arrays containing the processed data for training or test.
        """
        
        #set error sequence size to process later
        self.__error_sequence_size=error_sequence_size
        
        #set wavelet type 
        self.__wavelet_type=wavelet_type

        #check if data is sufficient for processing
        sorted_sampling_periods=np.sort(sampling_periods)[::-1]
        biggest_period=max(sampling_periods)
        required_timestamps=biggest_period*sequence_size

        if len(data)<required_timestamps:
            raise ValueError("Not enough data for processing. At least {} timestamps are required.".format(required_timestamps))

        if not (0 < test_ratio < 1):
            raise ValueError("Test ratio must be between 0 and 1.")
        

        #scale the data before processing
        data=self.scaler.scale1D(data)

        #build sequences w.r.t. parameters
        sequences= self._build_sequences(data, 
                                         sorted_sampling_periods, 
                                         sampling_statistics, 
                                         sequence_size, 
                                         wavelet_type, 
                                         required_timestamps)
        
        #normalize y then trim x and y to have same length
        y,mean_y=self.window_normalizer.normalize_with_prewindow(data, required_timestamps,step_size=1)
        y=np.array(y[required_timestamps:],dtype=np.float32) .reshape(-1,1)
        self.__final_y_processed=y
        self.__y_denormalization_term=np.array(mean_y[required_timestamps:],dtype=np.float32) .reshape(-1,1)
        X=sequences[:len(y)].astype(np.float32)

        #split train and test data --> TODO: the boundaries should be sharpened
        self.__test_size_index=int(len(X)*test_ratio)
        X_train, X_test = X[:-self.__test_size_index], X[-self.__test_size_index:]
        y_train, y_test = y[:-self.__test_size_index], y[-self.__test_size_index:]

        return X_train,X_test,y_train,y_test

    def preprocess_errors(self,errors: np.ndarray) -> List[np.ndarray]:
        """
        Performs operations on error data like windowization,normalization and wavelet transformation for error network

        Args:
            errors (np.ndarray): Transferred compensated errors.
        
        Returns:
            List[np.ndarray]: error values as x,y and denormalization term.
        """

        # windowization
        windowed_errors = self.build_windows(errors, window_length=self.__error_sequence_size) #sequence size and window size are same for error network

        # generates normalized X,y errors and denormalization term
        X=windowed_errors[:-1]
        X=self.window_normalizer.fit_transform(X)
        y,mean_y=self.window_normalizer.normalize_with_prewindow(errors, self.__error_sequence_size,step_size=1)
        y=np.array(y[self.__error_sequence_size:], dtype=np.float32).reshape(-1,1)
        denormalization_term=np.array(mean_y[self.__error_sequence_size:],dtype=np.float32).reshape(-1,1)

        # wavelet transform for X, return coeffs as numpy array
        X=np.apply_along_axis(self._calculate_dwt, 1, X,wavelet_type=self.__wavelet_type).astype(np.float32)

        return X,y,denormalization_term

    def _build_sequences(self,
                         data, 
                         sorted_sampling_periods, 
                         sampling_statistics, 
                         sequence_size, 
                         wavelet_type, 
                         max_window_size):
        """
            Builds sequences for the given data and given parameters.

            Args:
                data (np.ndarray): The data to be processed, as a NumPy array.
                sorted_sampling_periods (List[int]): List of sampling periods.
                sampling_statistics (List[str]): List of sampling statistics to be calculated, like 'mean'.
                sequence_size (int): The size of the sequence for processing.
                wavelet_type (str): Type of the wavelet to be used.
                max_window_size (int): The maximum window size.

            Returns:
                List[np.ndarray]: A list of NumPy arrays containing the processed data.
        """

        # builds data windows w.r.t. window size

        windowed_data = self.build_windows(data, window_length=max_window_size)

        # build sequences for each statistic with given periods and sequence size

        sequences = []

        for window in windowed_data:

            cascade_sequences = []

            for period in sorted_sampling_periods:
                
                stat_sequences = []

                for stat in sampling_statistics:
                
                    # if the statistics includes std, skip the last period (1-day) since it is a zero vector.
                    if stat == "std" and period == 1:
                        continue

                    # subsampled groups
                    groups = self._build_sampling_groups(window, period)
                
                    # calculate statistics for each group
                    groups_stats = [self._calculate_statistics(group, stat) for group in groups]

                    # trim the groups with the sequence size
                    groups_stats = np.array(groups_stats[-sequence_size:])

                    # normalization
                    normalized_groups= self.normalizer.fit_transform(groups_stats)
                    
                    # wavelet transform
                    wavelet_coeffs=self._calculate_dwt(normalized_groups, wavelet_type)

                    stat_sequences.append(wavelet_coeffs)
                
                cascade_sequences.append(stat_sequences)

            sequences.append(cascade_sequences)
            

        return np.array(sequences)

    def build_windows(self, values, window_length):

        """
            Builds past data windows for each time step.

            Args:
                values (List): The data to be processed, as a NumPy array.
                window_length (int): The element quantity included in the window.
            
            Returns:
                List[List]: A list of lists containing the past data windows for eeach timestamp.
        """
        windowed_data = []

        # for each timestamp, we need to build the past data
        for i in range(0, len(values)-window_length+1):
            window = values[i:i+window_length].flatten()
            windowed_data.append(window)

        return windowed_data

    def _build_sampling_groups(self, windowed_data, sampling_period):
        """
        Builds sampling groups for the given windowed data and sampling period.\
        For example, if the sampling period is 4, the first group will be the first 4 elements of the windowed data,\ 
        the second group will be the next 4 elements of the windowed data, and so on.
        
        Args:
             windowed_data (List): The windowed data to be sampled.
             sampling_period (int): The sampling period.
        Returns:
            List[List]: A list of lists containing the sampling groups.          
        """
        sampling_windows = []

        # Add full windows in reverse order
        for i in range(len(windowed_data), 0, -sampling_period):
            window = windowed_data[i-sampling_period:i]
            sampling_windows.append(window)

        # Add the last window with remaining elements
        remainder_size = len(windowed_data) % sampling_period
        if remainder_size > 0:
            last_window = windowed_data[:remainder_size]
            sampling_windows[-1] = last_window

        # reverse the order of windows
        sampling_windows = sampling_windows[::-1]
        
        return sampling_windows

    def _calculate_statistics(self,
                              data: List[float],
                              statistics_to_calculate: str) -> [int, float]:
        """
            Calculates the given statistics for the given data group.

            Args:
                data (List): The data group to be processed.
                statistics_to_calculate (str): The statistics to be performed on the given data group.

            Returns:
                [int, float]: The calculated statistics.    
        """

        # calculate statistics
        data = pd.Series(data)
        
        if statistics_to_calculate == "mean":
            return np.mean(data, axis=0)
        elif statistics_to_calculate == "std":
            return np.std(data, axis=0)
        elif statistics_to_calculate == "max":
            return np.max(data, axis=0)
        elif statistics_to_calculate == "min":
            return np.min(data, axis=0)
        elif statistics_to_calculate == "median":
            return np.median(data, axis=0)
        elif statistics_to_calculate == "mode":
            return data.mode()[0]
        elif statistics_to_calculate == "count":
            return len(data)
        elif statistics_to_calculate == "skew":
            return data.skew()
        elif statistics_to_calculate == "kurtosis":
            return data.kurtosis()
        else:
            raise ValueError(f"Unsupported statistics: {statistics_to_calculate}")
    
    def _calculate_dwt(self,
                        array: np.ndarray, 
                        wavelet_type: str,
                        level: int=None) -> List[float]:

        """ Returns the coefficients of the discrete wavelet transform of the given array except first coefficient.
            Args:
                array (np.ndarray): The array to be processed.
                wavelet_type (str): The type of the wavelet to be used.
                level (int, optional): The level of the wavelet transform. if None as default, possible max. level is used.
            
            Returns:
                List[float]: The coefficients of the discrete wavelet transform of the given array except first coefficient.
        """
    
       
        coeffs = wavedec(array, wavelet_type, mode="zero", level=level)      #default mode="zero"
       
        return np.concatenate(coeffs)[1:]
    
    def get_error_sequence_size(self):
        return self.__error_sequence_size
    
    def get_final_y_processed(self):
        """
        Returns: time shifted and size-trimmed y values for test and train data
        """
        if self.mode=="train":
            return self.__final_y_processed[self.__error_sequence_size+1:-self.__test_size_index]
        else:
            return self.__final_y_processed[-self.__test_size_index+self.__error_sequence_size+1:]

    
    def get_final_denormalization_term(self):
        """
        Returns: time shifted and size-trimmed y denormalization values for test and train data
        """        
        if self.mode=="train":
            return self.__y_denormalization_term[self.__error_sequence_size+1:-self.__test_size_index]
        else:  
            return self.__y_denormalization_term[-self.__test_size_index+self.__error_sequence_size+1:]
   
    def switch_mode(self,mode):
        self.mode=mode