from pecnet.utils import Utility
from pecnet.preprocessing import *
from pecnet.network import PecnetBuilder

"""A script example for pecnet framework."""

if __name__=='__main__':


    #loads default example dataset and plots it, close the plot and go on

    aapl_timestamps,aapl_prices=Utility.load_apple_test_dataset()

    Utility.plot(
        aapl_timestamps,
        aapl_prices,
        title='Apple Stock Prices',
        xlabel='Date',
        ylabel='Price ($)',
        tick_size=5,
        save_location=None)

    #preprocesses data and splits it into train and test sets
    X_train, X_test, y_train, y_test=DataPreprocessor().preprocess(data=aapl_prices[-1460:], # last 4 years
                                                                sampling_periods=[1,2,3],
                                                                sampling_statistics=["mean","std"],
                                                                sequence_size=4,
                                                                error_sequence_size=8,
                                                                wavelet_type="haar",
                                                                scale_factor=1.3, #1.2 etc.
                                                                input_normalization_type=None, #standard,minmax etc.
                                                                target_normalization_type="window_mean",
                                                                conjoincy=True,
                                                                test_ratio=0.05)

    # set seed for all components for reproducibility
    Utility.set_seed(42)

    #sets hyperparameters for pecnet framework, two way : manual or heuristic

    #Utility.set_hyperparameters(heuristic=True)
    #heuristic=True  means that
    # 1-)learning_rate=0.01
    # 2-)epoch_size=300
    # 3-)batch_size is square root of sample size
    # 4-)The number of hidden neurons in first layer is 2/3 the size of the input layer, plus the size of the output layer.
    # 5-)The number of hidden neurons in second layer is :
    # (sample_size/8*(input_sequence_size+output_sequence_size))-first_layer size and 8 is a scale factor, which can be changed.
    # There are 2 hidden layers in total. Because, there is currently no theoretical reason to use neural networks with any more than two hidden layers
    # or you can set hyperparameters manually by using Utility.set_hyperparameters() like below.

    Utility.set_hyperparameters(learning_rate=0.001,
                                epoch_size=500,
                                batch_size=96,
                                hidden_units_sizes=[16,8])

    #acts like fit() method

    pecnet = (PecnetBuilder().add_variable_network(X_train,y_train)
                                .add_error_network()
                                .add_final_network()
                                .build())

    #predictions for test set
    preds= pecnet.predict(X_test, test_target=y_test)

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


