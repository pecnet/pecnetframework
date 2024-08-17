import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Utility:

    heuristic=False

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
        timestamps = timestamps[-min_length:]
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
            
            if labels==None:
                label=f"Dataset {i}"
            else:
                label=labels[i-1]
            
            plt.plot(timestamps, dataset, markersize=2,linewidth=2.2,label=label)

        # Setting the xticks.if timesatamps are smaller than tick_size,all timestamps are plotted
        indices = np.linspace(0, len(timestamps) - 1, min(tick_size, len(timestamps)), dtype=int)
        plt.xticks(ticks=indices, labels=[timestamps[i] for i in indices])

        plt.legend(loc='upper left', fontsize=16,shadow=True)

        if save_location:
            plt.savefig(save_location,dpi=600,bbox_inches='tight')
            print(f"Plot saved to {save_location}")
        else:
            plt.show()

    @staticmethod
    def set_hyperparameters(heuristic=False,learning_rate=0.001,epoch_size=1000,batch_size=32,hidden_units_sizes=[32,16]):
        """
        Sets hyperparameters for pecnet framework.
        if you just set heuristic=True, that means:  
        1-)learning_rate=0.01 
        2-)epoch_size=300     
        3-)batch_size is square root of sample size
        4-)The number of hidden neurons in first layer is 2/3 the size of the input layer, plus the size of the output layer.
        5-)The number of hidden neurons in second layer is :\ 
        (sample_size/8*(input_sequence_size+output_sequence_size))-first_layer size and 8 is a scale factor, which can be changed.
        There are 2 hidden layers in total. Because, there is currently no theoretical reason to use neural networks with any more than two hidden layers
        
        """
        Utility.heuristic=heuristic
        Utility.learning_rate=learning_rate
        Utility.epoch_size=epoch_size
        Utility.batch_size=batch_size
        Utility.hidden_units_sizes=hidden_units_sizes
    
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

# Test code
if __name__ == '__main__':

    print(Utility.get_utility_path())
    
    timestamps, values = Utility.load_apple_test_dataset()
    print("Timestamps:", timestamps)
    print("Values:", values)

    # Call the static plot method
    Utility.plot(timestamps, values, title='Apple Stock Prices', xlabel='Date', ylabel='Price ($)',save_location='C:\\Users\\srknm\\Desktop\\plot.jpg')


