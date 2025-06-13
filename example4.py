from pecnet.utils import Utility
from pecnet.preprocessing import *
from pecnet.network import PecnetBuilder

import pandas as pd

# Load data
df = pd.read_csv("pecnet/example_datasets/5G_ProcessedData.csv")

# Define columns to process
columns_to_process = [
    'NR_Scan_SSB_RSRP_SortedBy_RSRP_merged_Scanner',
    'NR_UE_RSRP_0_DL',
    'NR_UE_RSRP_0_UL',
    'NR_Scan_SSB_RSRP_SortedBy_RSRP_diff_0_1_Scanner'
]

target_column = 'Easting'

if __name__ == '__main__':

    preprocessed_inputs_train=[]
    preprocessed_inputs_test=[]

    target_series = df[target_column].dropna().values

    # Preprocess target first → this will define DataPreprocessor().target_*
    X_train_0, X_test_0, y_train, y_test = DataPreprocessor().preprocess(
        data=target_series,
        sampling_periods=[1, 2],
        sampling_statistics=["mean", "std"],
        sequence_size=4,
        error_sequence_size=8,
        wavelet_type="haar",
        scale_factor=1.3,
        input_normalization_type=None,
        target_normalization_type="window_mean",
        conjoincy=False,
        test_ratio=0.05
    )
    preprocessed_inputs_train.append(X_train_0)
    preprocessed_inputs_test.append(X_test_0)

    for col in columns_to_process:
        series = df[col].dropna().values

        X_train, X_test, _, _ = DataPreprocessor().preprocess(
            data=series,
            sampling_periods=[1,2,3],
            sampling_statistics=["mean","std"],
            sequence_size=4,
            error_sequence_size=8,
            wavelet_type="haar",
            scale_factor=1.3,
            input_normalization_type=None,
            target_normalization_type="window_mean",
            conjoincy=False,
            test_ratio=0.05
        )

        preprocessed_inputs_train.append(X_train)
        preprocessed_inputs_test.append(X_test)

    # Set hyperparameters
    Utility.set_hyperparameters(
        learning_rate=0.001,
        epoch_size=100,
        batch_size=96,
        hidden_units_sizes=[16,32,16,8]
    )

    # Build Pecnet model
    pecnet = PecnetBuilder()

    for i, X_train_input in enumerate(preprocessed_inputs_train):
        if i == 0:
            pecnet.add_variable_network(X_train_input, y_train)
        else:
            pecnet.add_variable_network(X_train_input)

    # Add ErrorNetwork and FinalNetwork
    pecnet = (pecnet.add_error_network()
                    .add_final_network()
                    .build())

    # Predictions for test set
    preds = pecnet.predict(*preprocessed_inputs_test, test_target=y_test)

    # Tomorrow's prediction
    print("Tomorrow's prediction: ", preds[-1])

    # Evaluate
    result = pecnet.evaluate(preds, target_series)
    print(result)

    # Optional: plot (if you want to visualize)
    Utility.plot(
        list(range(len(target_series))),
        target_series,
        preds[:-1],
        title='5G Target Predictions',
        xlabel='Sample',
        ylabel='Target Value',
        tick_size=5,
        labels=["Actual","Predicted"],
        save_location=None
    )
