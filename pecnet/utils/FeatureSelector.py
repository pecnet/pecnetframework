import numpy as np

class FeatureSelector:
    """
    FeatureSelector ranks candidate input features by their correlation with a given reference vector.
    It supports both the initial feature selection using the target and subsequent selections using error outputs.
    """

    def __init__(self, threshold=0.25):
        """
        Args:
            threshold (float): Minimum absolute correlation required to select a feature.
        """
        self.threshold = threshold
        self.selected_indices = []
        self.last_correlation = None
        self.correlation_log = []  # list of (selected_index, correlation)

    def _flatten(self, feature_array):
        """
        Flattens the feature arrays by taking the last time step (most recent) across all samples.

        Args:
            feature_array (np.ndarray): Feature array of shape (n_samples, ..., time)

        Returns:
            np.ndarray: Flattened array of shape (n_samples,)
        """
        return feature_array.reshape(feature_array.shape[0], -1)[:, 0] #low frequency component

    def _compute_correlations(self, candidates, reference):
        """
        Computes Pearson correlation between each candidate feature and the reference values.

        Args:
            candidates (List[np.ndarray]): List of 1D arrays.
            reference (np.ndarray): 1D reference values.

        Returns:
            np.ndarray: Correlation values.
        """
        return np.array([np.corrcoef(f, reference)[0, 1] for f in candidates])

    def select_next(self, feature_list, reference_vector, force_include_best_if_first=False):
        """
        Selects the next best feature based on correlation with the given reference values.

        Args:
            feature_list (List[np.ndarray]): List of input feature arrays (train sets).
            reference_vector (np.ndarray): The vector to compare against (target or error).
            force_include_best_if_first (bool): If True, disables threshold check for first feature.

        Returns:
            int or None: Index of selected feature, or None if none satisfy the threshold.
        """
        reference_vector = reference_vector.reshape(-1)

        candidates = [
            i for i in range(len(feature_list)) if i not in self.selected_indices
        ]
        if not candidates:
            return None

        flattened = [
            self._flatten(feature_list[i]) for i in candidates
        ]
        correlations = self._compute_correlations(flattened, reference_vector)

        max_idx = int(np.argmax(np.abs(correlations)))
        best_corr = correlations[max_idx]

        if abs(best_corr) < self.threshold and not (force_include_best_if_first and not self.selected_indices):
            return None

        selected_index = candidates[max_idx]
        self.selected_indices.append(selected_index)

        self.last_correlation = best_corr
        self.correlation_log.append((selected_index, best_corr))

        return selected_index

    def get_last_corr_score(self):
        return self.last_correlation

    def get_all_corr_scores(self):
        return self.correlation_log