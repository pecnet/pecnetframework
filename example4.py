from pecnet.utils import Utility
from pecnet.preprocessing import *
from pecnet.network import PecnetBuilder
from pecnet.utils import FeatureSelector

# Load data
df = pd.read_csv("pecnet/example_datasets/5G_ProcessedData.csv")

# Define columns to process
feature_columns = [
    'NR_Scan_SSB_RSRP_SortedBy_RSRP_merged_Scanner',
    'NR_UE_RSRP_0_DL',
    'NR_UE_RSRP_0_UL',
    'NR_Scan_SSB_RSRP_SortedBy_RSRP_diff_0_1_Scanner'
]

target_column = 'Northing'

if __name__ == '__main__':

    target_series = np.array(df[target_column].dropna().values)

    # Preprocess target first
    X_train_0, X_test_0, y_train, y_test = DataPreprocessor().preprocess(
        data=target_series,
        sampling_periods=[1, 2,3],
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

    candidate_X_train = [X_train_0]
    candidate_X_test = [X_test_0]

    for col in feature_columns:

        series = np.array(df[col].dropna().values)

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

        candidate_X_train.append(X_train)
        candidate_X_test.append(X_test)

    # Set seed
    Utility.set_seed(42)

    # Set hyperparameters
    Utility.set_hyperparameters(
        learning_rate=0.001,
        epoch_size=150,
        batch_size=96,
        hidden_units_sizes=[16,32,16,8]
    )

    # Build Pecnet model
    pecnet = PecnetBuilder()
    selector = FeatureSelector(threshold=0.08)


    # Step 1: Set orig.target's network with most correlated feature.that is highly probable its past samples
    # Step 2: Iterative correlation-based feature selection

    total_columns=[target_column]+feature_columns
    target_reference = y_train  # initial reference is the target itself
    initial_network_setting = True

    while True:
        idx = selector.select_next(candidate_X_train, target_reference, force_include_best_if_first=initial_network_setting)
        if idx is None:
            print("No more features passed correlation threshold.")
            break

        pecnet.add_variable_network(candidate_X_train[idx], y_train if initial_network_setting else target_reference)
        print(f"Added feature: {total_columns[idx]} | corr: {selector.get_last_corr_score():.3f}")

        target_reference = pecnet.pecnet.get_next_variable_network_target_values()
        initial_network_setting = False

    # Add ErrorNetwork and FinalNetwork
    pecnet = (pecnet.add_error_network()
                    .add_final_network()
                    .build())

    # Select test features in same order
    selected_X_test = [candidate_X_test[i] for i in selector.selected_indices]
    test_inputs = [X_test_0] + selected_X_test

    # Predictions for test set
    preds = pecnet.predict(*test_inputs, test_target=y_test)

    # Tomorrow's prediction
    print("Last predicted value:", preds[-1])

    # Evaluate
    result = pecnet.evaluate(preds, target_series)
    print(result)

    # Plot results
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

