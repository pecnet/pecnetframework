from typing import List, Any
from copy import deepcopy
from numpy import ndarray, dtype
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.stride_tricks import as_strided
from pywt import wavedec
import pandas as pd
from pecnet.preprocessing.Normalizers import *
from typing import Sequence

class DataPreprocessor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataPreprocessor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):

        if not hasattr(self, 'initialized'): # Check if instance is already initialized
            print("DataPreprocessor Initialized")
            self.initialized = True
            self.mode="train"
            self.scaler=Scaler()
            self.normalizer = None
            self.window_normalizer=WindowNormalizer()
            self.mean_normalizer=MeanNormalizer()
            self.__target_normalization_step_size = 1

            self.__final_y_processed = None
            self.__sequence_size = None
            self.__error_sequence_size = None
            self.__y_denormalization_term = None
            self.__wavelet_type = None
            self.__required_timestamps = None
            self.__test_size_index = None

            self.target = None
            self.target_denormalization_term = None
            self.target_scaler = None
            self.target_normalizer = None

    def preprocess(self,
                   data: np.ndarray, 
                   sampling_periods:List[int] = None,
                   stride : int = None,
                   sampling_statistics:List[str] = None,
                   sequence_size:int = 4, 
                   error_sequence_size:int = 4, 
                   wavelet_type:str ="haar",
                   scale_factor=None,
                   input_normalization_type=None,
                   target_normalization_type="window_mean",
                   conjoincy=False,
                   test_ratio:float = 0.2
                   ) -> tuple[
        ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
            Wrapper method --> Preprocesses the given data using wavelet transform and statistical analysis.

            Args:
                data (np.ndarray): The data to be processed, as a NumPy array.
                sampling_periods (List[int], optional): List of sampling periods. Defaults to [1, 4].
                stride (int): sequence jump size while downsampling.
                sampling_statistics (List[str], optional): List of sampling statistics to be calculated, like 'mean'. Defaults to ["mean"].
                sequence_size (int, optional): The size of the sequence for processing. Defaults to 4.
                error_sequence_size (int, optional): The size of the sequence for error network. Defaults to 4.
                wavelet_type (str, optional): Type of the wavelet to be used. Defaults to "haar".
                scale_factor (float, optional): Scaling factor. Defaults to None.
                input_normalization_type (str, optional): Input normalization type like min-max, standard. Defaults to None.
                target_normalization_type (str, optional): Target normalization type like window-mean, ema. Defaults to window-mean.
                conjoincy (bool, optional): Conjunct or not. Defaults to False. It defines the way of divide-slide operation on the data.
                test_ratio (float, optional): The ratio of the test data. Defaults to 0.2.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of NumPy arrays containing the processed data for training or test.
        """

        #assings default values for mutable variables
        sampling_periods = sampling_periods or [1, 4]
        sampling_statistics = sampling_statistics or ["mean"]

        #set sequence and error sequence size to process later
        self.__error_sequence_size=error_sequence_size
        self.__sequence_size=sequence_size
        
        #set wavelet type 
        self.__wavelet_type=wavelet_type

        #check if data is sufficient for processing
        sorted_sampling_periods=np.sort(sampling_periods)[::-1].tolist()
        biggest_period=max(sampling_periods)

        if conjoincy:
            required_timestamps=biggest_period+sequence_size-1
        else:
            required_timestamps=biggest_period*sequence_size

        self.__required_timestamps=required_timestamps

        if len(data)<required_timestamps:
            raise ValueError("Not enough data for processing. At least {} timestamps are required.".format(required_timestamps))

        if not (0 < test_ratio < 1):
            raise ValueError("Test ratio must be between 0 and 1.")
        

        #scaling and normalization options
        split_index=int(test_ratio*(len(data[:-required_timestamps])-1)) #"required timestamps" for windowization, "-1" for y.
        train_part = data[:-split_index]
        test_part = data[-split_index:]

        #scale the data before processing
        if scale_factor is not None:
            train_part=self.scaler.fit_scale1D(train_part,scale_factor)
            test_part=self.scaler.scale1D(test_part)

            if self.target_scaler is None:
                self.target_scaler = deepcopy(self.scaler)

        #TODO:expand for other normalizations
        #init normalizer if normalization is required
        if input_normalization_type is None:
            self.normalizer=None
        else:
            self.normalizer=Normalizer(input_normalization_type)
            if self.target_normalizer is None:
                self.target_normalizer = deepcopy(self.normalizer)
            
            # Train ve test verilerini normalize et
            train_part = self.normalizer.fit_transform(train_part.reshape(-1,1))
            test_part = self.normalizer.transform(test_part.reshape(-1,1))

        # Train ve test verilerini tekrar birleştir ve full data 'yı oluştur
        data = np.concatenate([train_part, test_part])
        raw_data = np.copy(data)

        #build sequences w.r.t. parameters
        print("Initial data size:",len(data), "------")
        if conjoincy:
            sequences= self._build_conjoined_sequences(data, 
                                         sorted_sampling_periods, 
                                         sampling_statistics, 
                                         sequence_size, 
                                         wavelet_type)
        else:
            sequences= self._build_sequences(data, 
                                         sorted_sampling_periods, 
                                         sampling_statistics, 
                                         sequence_size,
                                         stride,
                                         wavelet_type, 
                                         required_timestamps)


        #normalize y with window_mean then adjust x and y to have same length
        if target_normalization_type == "window_mean":
            y, mean_y = self.window_normalizer.normalize_with_prewindow(
                raw_data, required_timestamps, step_size=self.__target_normalization_step_size)
            y = np.array(y[required_timestamps:], dtype=np.float32)
            mean_y = np.array(mean_y[required_timestamps:], dtype=np.float32)
        elif target_normalization_type == "ema":
            y, mean_y = self.window_normalizer.normalize_with_ema(
                raw_data, required_timestamps, step_size=self.__target_normalization_step_size)
            y = np.array(y[required_timestamps:], dtype=np.float32)
            mean_y = np.array(mean_y[required_timestamps:], dtype=np.float32)
        elif target_normalization_type is None:
            y = np.array(raw_data[required_timestamps:], dtype=np.float32)
            mean_y = np.zeros_like(y)
        else:
            raise ValueError(f"Unsupported target_normalization_type: {target_normalization_type}")
        print("X: ",len(sequences), "Y: ",len(y),"---before adjustment",)
        self.__final_y_processed=np.append(y,0).reshape(-1,1) # add a zero to the end as a placeholder for the tomorrow's prediction
        self.__y_denormalization_term= np.append(mean_y,0).reshape(-1,1) # add a zero mean to the end as a placeholder for the tomorrow's prediction

        if self.target is None:
            self.target = self.__final_y_processed.copy()

        if self.target_denormalization_term is None:
            self.target_denormalization_term = self.__y_denormalization_term.copy()

        X=np.asarray(sequences[:len(self.target)]) #includes tomorrow prediction as placeholder

        #split train and test data
        self.__test_size_index=int(len(X)*test_ratio) #TODO: there is also split_index, make it one variable
        X_train, X_test = X[:-self.__test_size_index], X[-self.__test_size_index:]
        y_train, y_test = self.__final_y_processed[:-self.__test_size_index], self.__final_y_processed[-self.__test_size_index:]

        return X_train,X_test,y_train,y_test # only main (first) network's used as y.

    def preprocess_errors(self,errors: np.ndarray) -> List[np.ndarray]:
        """
        Performs operations on error data:  alignment, windowization,normalization and wavelet transformation for error network

        Args:
            errors (np.ndarray): Transferred compensated errors.
        
        Returns:
            List[np.ndarray]: error values as x,y and denormalization term.
        """

        # windowization
        windowed_errors = self.build_windows(errors, window_length=self.__error_sequence_size) # window size is treated as sequence size in error network.

        # normalized and wavelet transformed X
        X=windowed_errors[:-1]
        X=self.window_normalizer.fit_transform(X) # X will be training or test set not whole.So it is secure to do so.
        X_errors=np.apply_along_axis(self._calculate_dwt, 1, X.copy(),wavelet_type=self.__wavelet_type).astype(np.float32)

        # y will be kept same as before
        y_errors = np.array(errors[self.__error_sequence_size:], dtype=np.float32).reshape(-1, 1)

        return X_errors,y_errors

    def _build_sequences(self,
                         data, 
                         sorted_sampling_periods, 
                         sampling_statistics, 
                         sequence_size,
                         stride,
                         wavelet_type, 
                         max_window_size):
        """
            Builds sequences for the given data and given parameters.

            Args:
                data (np.ndarray): The data to be processed, as a NumPy array.
                sorted_sampling_periods (np.ndarray of int): List of sampling periods.
                sampling_statistics (List[str]): List of sampling statistics to be calculated, like 'mean'.
                sequence_size (int): The size of the sequence for processing.
                wavelet_type (str): Type of the wavelet to be used.
                max_window_size (int): The maximum window size.

            Returns:
                List[np.ndarray]: A list of NumPy arrays containing the processed data.
        """

        # builds data windows w.r.t. window size

        windowed_data = self.build_windows(data, window_length=max_window_size,stride=stride)

        # build sequences for each statistic with given periods and sequence size

        sequences = []

        for window in windowed_data:

            cascade_sequences = []

            for period in sorted_sampling_periods:
                
                stat_sequences = []

                for stat in sampling_statistics:
                    
                    #That case will be controlled in the variable network
                    # # if the statistics includes std, skip the last period (1-day) since it is a zero vector.
                    # if stat == "std" and period == 1:
                    #     continue

                    # subsampled groups
                    groups = self._build_sampling_groups(window, period)
                
                    # calculate statistics for each group
                    groups_stats = [self._calculate_statistics(group, stat) for group in groups]

                    # trim the groups with the sequence size
                    groups_stats = np.array(groups_stats[-sequence_size:])

                    # normalization
                    normalized_groups= self.mean_normalizer.fit_transform(groups_stats)
                    
                    # wavelet transform
                    wavelet_coeffs=self._calculate_dwt(normalized_groups, wavelet_type)
                    
                    stat_sequences.append(wavelet_coeffs)
                
                cascade_sequences.append(stat_sequences)

            sequences.append(cascade_sequences)    

        return np.array(sequences)

    def _build_conjoined_sequences(self,
                         data, 
                         sorted_sampling_periods, 
                         sampling_statistics, 
                         sequence_size, 
                         wavelet_type):
        """
            Builds sequences for the given data and given parameters.

            Args:
                data (np.ndarray): The data to be processed, as a NumPy array.
                sorted_sampling_periods (List[int]): List of sampling periods.
                sampling_statistics (List[str]): List of sampling statistics to be calculated, like 'mean'.
                sequence_size (int): The size of the sequence for processing.
                wavelet_type (str): Type of the wavelet to be used.

            Returns:
                List[np.ndarray]: A list of NumPy arrays containing the processed data.
        """
        sequences = []
        max_period = sorted_sampling_periods[0]

        for period in sorted_sampling_periods:

            stat_sequences = []
            sliding_windows=self.build_windows(data[max_period-period:], window_length=period)

            for stat in sampling_statistics:
                
                window_sequences = []

                for i in range(0,len(sliding_windows)-sequence_size+1):

                    #That case below will be controlled in the variable network
                    # # if the statistics includes std, skip the last period (1-day) since it is a zero vector.
                    # if stat == "std" and period == 1:
                    #     continue

                    # window sequences
                    groups = sliding_windows[i:i+sequence_size]
                
                    # calculate statistics for each group
                    groups_stats = [self._calculate_statistics(group, stat) for group in groups]

                    # trim the groups with the sequence size
                    groups_stats = np.array(groups_stats[-sequence_size:])

                    # normalization
                    normalized_groups= self.mean_normalizer.fit_transform(groups_stats)
                    
                    # wavelet transform
                    wavelet_coeffs=self._calculate_dwt(normalized_groups, wavelet_type)
                    
                    window_sequences.append(wavelet_coeffs)
                
                stat_sequences.append(window_sequences)

            sequences.append(stat_sequences)   

        sequences = np.array(sequences)
        sequences=sequences.transpose(2, 0, 1, 3) #axes were adjusted same as build_windows method
        return sequences

    def build_windows(self,values: Sequence[float], window_length: int, stride: int = None) -> np.ndarray:
        """
        Splits a 1D sequence into overlapping or strided sliding windows of fixed length.
        Parameters
        ----------
        values : Sequence[float]
            1D input sequence (list, tuple, or np.ndarray).
        window_length : int
            Length of each sliding window.
        stride (int, optional): Stride between windows. Defaults to 1 (fully overlapping).

        Returns
        -------
        np.ndarray
            2D array of shape (n_windows, window_length), or empty if not enough data.

        Example
        -------
        >>> build_windows([1, 2, 3, 4], 2)
        array([[1, 2],
               [2, 3],
               [3, 4]])
        """
        values = np.asarray(values)
        if values.ndim == 2:
            values = values.squeeze()
        if values.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {values.shape}")

        if len(values) < window_length:
            return np.empty((0, window_length))  # to keep shape adjusted

        if stride is None or stride == 1:
            return sliding_window_view(values, window_length).reshape(-1, window_length)

        # if stride>1 is given
        n = (len(values) - window_length) // stride + 1
        if n <= 0:
            return np.empty((0, window_length))

        return as_strided(values,
                          shape=(n, window_length),
                          strides=(values.strides[0] * stride, values.strides[0]))

    def _build_sampling_groups(self, windowed_data, sampling_period):
        """
        Builds sampling groups from a 2D windowed data array based on the given sampling period.

        This function emulates the original behavior:
        - It collects backward groups of size `sampling_period`,
        - If a remaining group exists that doesn't fill the full window, it replaces the last group with it,
        - Finally, it reverses the list to maintain chronological order.

        Args:
            windowed_data (List[List] or np.ndarray): The input 2D array containing time windowed values.
            sampling_period (int): Number of elements in each group.

        Returns:
            List[np.ndarray]: A list of grouped samples (each as a 2D array).
        """
        data = np.asarray(windowed_data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        groups = []
        i = len(data)
        use_ceil = True if isinstance(sampling_period, float) else False

        # Iterate backwards with stride = sampling_period
        while i > 0:
            group_size = int(np.ceil(sampling_period)) if use_ceil else int(sampling_period)
            start_idx = max(i - group_size, 0)
            group = data[start_idx:i]
            groups.append(group)

            i -= group_size
            if isinstance(sampling_period, float):
                use_ceil = not use_ceil  # alternate between ceil and floor

        # Replace last group with remainder if needed
        remainder_size = len(data) % sampling_period
        if remainder_size > 0:
            groups[-1] = data[:remainder_size]

        # Reverse to restore chronological order
        groups.reverse()

        return groups

    def _calculate_statistics(self,
                              data: np.ndarray,
                              statistics_to_calculate: str) -> [int, float]:
        """
            Calculates the given statistics for the given data group.

            Args:
                data (List): The data group to be processed.
                statistics_to_calculate (str): The statistics to be performed on the given data group.

            Returns:
                [int, float]: The calculated statistics.    
        """

        #calculate statistics
        data = pd.Series(data.flatten())
        
        if statistics_to_calculate == "mean":
            return np.nanmean(data, axis=0)
        elif statistics_to_calculate == "std":
            return np.nanstd(data, axis=0)
        elif statistics_to_calculate == "max":
            return np.nanmax(data, axis=0)
        elif statistics_to_calculate == "min":
            return np.nanmin(data, axis=0)
        elif statistics_to_calculate == "median":
            return np.nanmedian(data, axis=0)
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
            return self.target[self.__error_sequence_size+1:-self.__test_size_index]
        else:
            return self.target[-self.__test_size_index+self.__error_sequence_size+1:]

    
    def get_final_denormalization_term(self):
        """
        Returns: time shifted and size-trimmed y denormalization values for test and train data
        Explanation:
        - The offset (error_sequence_size + 1) is windowing offset + 1 unit time shift
        """        
        if self.mode=="train":
            return self.target_denormalization_term[self.__error_sequence_size+1:-self.__test_size_index]
        else:  
            return self.target_denormalization_term[-self.__test_size_index+self.__error_sequence_size+1:]
   
    def generate_final_denormalization_term_for_last_pred_element(self):
        
        raw_targets=self.target+self.target_denormalization_term
        norm_term=np.mean(raw_targets[-(self.__required_timestamps+self.__target_normalization_step_size):-self.__target_normalization_step_size].flatten())
        
        return norm_term.reshape(-1,1)

    def switch_mode(self,mode):
        self.mode=mode

    def reset(self):
        """Resets the state of the singleton instance to its initial values."""
        print("DataPreprocessor Reset")
        self.initialized = True
        self.mode = "train"
        self.scaler = Scaler()
        self.normalizer = None
        self.window_normalizer = WindowNormalizer()
        self.mean_normalizer = MeanNormalizer()
        self.__target_normalization_step_size = 1

        self.__final_y_processed = None
        self.__sequence_size = None
        self.__error_sequence_size = None
        self.__y_denormalization_term = None
        self.__wavelet_type = None
        self.__required_timestamps = None
        self.__test_size_index = None

        self.target = None
        self.target_denormalization_term = None
        self.target_scaler = None
        self.target_normalizer = None