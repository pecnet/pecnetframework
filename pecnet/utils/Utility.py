import os
import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

class Utility:

    heuristic=False

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Sets global seed for NumPy, Python, PyTorch and OS hash to ensure reproducibility.

        Args:
            seed (int): The random seed to set.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_hyperparameters(heuristic=False,learning_rate=0.001,epoch_size=1000,batch_size=32,hidden_units_sizes=[32,16]):
        """
        Sets hyperparameters for pecnet framework.
        if you just set heuristic=True, that means:
        1-)learning_rate=0.01
        2-)epoch_size=300
        3-)batch_size is square root of sample size
        4-)The number of hidden neurons in first layer is 2/3 the size of the input layer, plus the size of the output layer.
        5-)The number of hidden neurons in second layer is :
        (sample_size/8*(input_sequence_size+output_sequence_size))-first_layer size and 8 is a scale factor, which can be changed.
        There are 2 hidden layers in total. Because, there is currently no theoretical reason to use neural networks with any more than two hidden layers
        """
        Utility.heuristic=heuristic
        Utility.learning_rate=learning_rate
        Utility.epoch_size=epoch_size
        Utility.batch_size=batch_size
        Utility.hidden_units_sizes=hidden_units_sizes

    @staticmethod
    def get_utility_path():
        """
        Returns the directory in which the current file (Utility.py) is located.
        """
        return os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def load_apple_test_dataset():
        """
        Reads a dataset from the default apple stock prices .csv file path and returns the values and timestamps as NumPy arrays.
        """

        file_path = Utility.get_utility_path()+'/../example_datasets/apple_stock_prices.csv'
        df = pd.read_csv(file_path)

        timestamps = np.array(df['Date'])
        values = np.array(df['Adj Close'])
        return timestamps, values


    @staticmethod
    def plot(timestamps, *datasets, title='Data', xlabel='Date', ylabel='Y-axis',tick_size=5,labels=None,save_location=None):

        """
        Plots the given datasets with respect to the given timestamps and plot parameters.
        """

        # Trimming the datasets to the same length w.rt. predictions
        min_length = min(len(dataset) for dataset in datasets)
        timestamps = pd.to_datetime(timestamps[-min_length:],dayfirst=True)
        datasets = [dataset[-min_length:] for dataset in datasets]

        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20
        }

        plt.figure(figsize=(20,5))
        plt.title(title, fontdict=font)
        plt.xlabel(xlabel, fontsize=17)
        plt.ylabel(ylabel, fontsize=17)
        plt.tick_params(labelcolor='0', labelsize='15', width=5)
        plt.grid(True, linestyle='-.')


        # Plotting the datasets
        for i, dataset in enumerate(datasets, start=1):
            
            label = f"Dataset {i}" if labels is None else labels[i - 1]
            plt.plot(timestamps, dataset, markersize=2,linewidth=2.2,label=label)

        # Setting the xticks.if timesatamps are smaller than tick_size,all timestamps are plotted
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

        if tick_size > 0:
            step = max(1, len(timestamps) // tick_size)
            plt.xticks(ticks=timestamps[::step])

        plt.legend(loc='upper left', fontsize=16,shadow=True)
        plt.tight_layout()

        if save_location:
            plt.savefig(save_location,dpi=600,bbox_inches='tight')
            print(f"Plot saved to {save_location}")
        else:
            plt.show()

    
    @staticmethod
    def make_pct(data):
        
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        pct_change = data.pct_change().fillna(0)
        pct_change.iloc[0] = 0

        return pct_change.to_numpy()
    
    @staticmethod
    def convert_pct_back(data,initial_value):
        
        converted_data = [initial_value]
        for pct in data[1:]:
            new_data = converted_data[-1] * (1 + pct)
            converted_data.append(new_data)
        
        return np.array(converted_data)

    @staticmethod
    def convert_pct_preds_back(preds,reals): #reals should be 1 step back in time at same index
        
        converted_preds = []
        for i in range (0,len(preds)):
            new_value = reals[i] * (1 + preds[i])
            converted_preds.append(new_value)
        
        return np.array(converted_preds)

    @staticmethod
    def create_difference_series(time_series):

        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        difference_series = [0]
        
        for i in range(1, len(time_series)):
            difference = time_series[i] - time_series[i - 1]
            difference_series.append(difference)
        
        return np.array(difference_series)

    @staticmethod
    def convert_diffs_to_origin(difference_series, initial_value):
        # Initialize the original series with the first element as 0
        original_series = [initial_value]
        
        # Calculate the cumulative sum to obtain the original series
        for i in range(1, len(difference_series)):
            original_value = original_series[i - 1] + difference_series[i]
            original_series.append(original_value)
        
        return np.array(original_series)

    @staticmethod
    def convert_diffs_preds_to_origin(preds,reals): #reals should be 1 step back in time at same index

        original_series = []
        
        # Calculate the cumulative sum to obtain the original series
        for i in range(0, len(preds)):
            original_value = preds[i] + reals[i]
            original_series.append(original_value)
    
        return np.array(original_series)

    @staticmethod
    def dataframe_to_dict(dataframe, time_column='Date'):
        """
        Converts a multivariate pandas DataFrame into a dictionary with column names as keys and their values as NumPy arrays.
        Separates the time or date column into its own array.

        Parameters:
        dataframe (pd.DataFrame): The multivariate data to be converted.
        time_column (str): The name of the time or date column.

        Returns:
        dict: A dictionary with keys as column names and values as NumPy arrays.
        np.ndarray: A NumPy array containing time or date values.
        """
        if time_column in dataframe.columns:
            time_values = dataframe.pop(time_column).to_numpy()
        elif dataframe.index.name == time_column:
            time_values = dataframe.index.to_numpy()
        else:
            raise ValueError(f"Time column '{time_column}' not found in the dataframe or as an index.")

        data_dict = {col: dataframe[col].to_numpy() for col in dataframe.columns}

        return time_values, data_dict

# Test code
if __name__ == '__main__':

    # print(Utility.get_utility_path())
    #
    # timestamps, values = Utility.load_apple_test_dataset()
    # print("Timestamps:", timestamps)
    # print("Values:", values)
    #
    # # Call the static plot method
    # Utility.plot(timestamps, values, title='Apple Stock Prices', xlabel='Date', ylabel='Price ($)',save_location='C:\\Users\\srknm\\Desktop\\plot.jpg')

    # Example of converting a DataFrame to a dictionary
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=5, freq='D'),
        'Open': [300, 305, 310, 308, 307],
        'Close': [305, 310, 308, 307, 312],
        'Volume': [1000, 1100, 1050, 1075, 1200]
    }
    df = pd.DataFrame(data)

    time_values, data_dict = Utility.dataframe_to_dict(df, time_column='Date')
    print("Time Values:", time_values)
    print("Data Dictionary:", data_dict)

