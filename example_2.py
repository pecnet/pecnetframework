from pecnet.utils import Utility
from pecnet.preprocessing import *
from pecnet.network import PecnetBuilder
import yfinance as yf

"""A multivariate script example for pecnet framework."""

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date,auto_adjust=False)
    data.index.name = "Date"
    data.columns = data.columns.droplevel('Ticker')
    return data

if __name__ == '__main__':

    stock_dataframe=get_stock_data('AAPL','2020-01-01','2025-01-02')
    nasdaq100_dataframe=get_stock_data('^NDX','2020-01-01','2025-01-02')

    aapl_timestamps,aapl=Utility.dataframe_to_dict(stock_dataframe,time_column='Date')
    _, nasdaq100index = Utility.dataframe_to_dict(nasdaq100_dataframe, time_column='Date')

    aapl_prices=aapl['Close']
    aapl_volume=aapl['Volume']
    nasdaq100index_prices=nasdaq100index['Close']

    Utility.plot(
        aapl_timestamps,
        aapl_prices,
        title='Apple Stock Prices',
        xlabel='Date',
        ylabel='Price ($)',
        tick_size=10,
        save_location=None)

    #preprocesses data and splits it into train and test sets
    X_train, X_test, y_train, y_test=DataPreprocessor().preprocess(data=aapl_prices,
                                                                sampling_periods=[1,2,4],
                                                                sampling_statistics=["mean","std"],
                                                                sequence_size=4,
                                                                error_sequence_size=8,
                                                                wavelet_type="haar",
                                                                scale_factor=1.3, #1.2 etc.
                                                                input_normalization_type=None, #standard,minmax etc.
                                                                target_normalization_type="window_mean",
                                                                conjoincy=False,
                                                                test_ratio=0.05)

    X_train_index, X_test_index, _, _=DataPreprocessor().preprocess(data=nasdaq100index_prices,
                                                                sampling_periods=[1,2,4],
                                                                sampling_statistics=["mean","std"],
                                                                sequence_size=4,
                                                                error_sequence_size=8,
                                                                wavelet_type="haar",
                                                                scale_factor=1.3, #1.2 etc.
                                                                input_normalization_type=None, #standard,minmax etc.
                                                                target_normalization_type="window_mean", #window_mean, ema etc.
                                                                conjoincy=False,
                                                                test_ratio=0.05)
    # set a seed for reproducibility
    Utility.set_seed(42)

    # #sets hyperparameters for pecnet framework
    Utility.set_hyperparameters(learning_rate=0.001,
                                epoch_size=400,
                                batch_size=96,
                                hidden_units_sizes=[32,64,32,16])

    #acts like fit() method

    pecnet = (PecnetBuilder().add_variable_network(X_train,y_train)
                                .add_variable_network(X_train_index)
                                .add_error_network()
                                .add_final_network()
                                .build())


    #predictions for test set

    preds= pecnet.predict(X_test, X_test_index, test_target=y_test)

    #tomorrow's prediction
    print("Tomorrow's prediction: ",preds[-1])

    #Evaluates results in terms of RMSE,MAPE,R2.

    result=pecnet.evaluate(preds, aapl_prices)
    print(result)


    #plot predictions to compare with ground truths

    Utility.plot(
        aapl_timestamps,
        aapl_prices,
        preds[:-1],
        title='Apple Stock Prices',
        xlabel='Date',
        ylabel='Price ($)',
        tick_size=5,
        labels=["Actual","Predicted"],
        save_location=None)


