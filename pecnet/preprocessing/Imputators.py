import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class Imputator:
    """
    Imputation class for 1D arrays with missing values.
    Supports both x-missing and y-missing scenarios using different tools.(local GP modeling, etc.)
    """

    @staticmethod
    def impute_with_gp(x: np.ndarray, y: np.ndarray, target: str = "y", h: int = 2) -> np.ndarray:
        """
        Imputes missing values (None or np.nan) in x or y using Gaussian Process regression.

        Args:
            x (np.ndarray): 1D array of x-values, with possible missing values (if target='x').
            y (np.ndarray): 1D array of y-values, with possible missing values (if target='y').
            target (str): Which array to impute: 'x' or 'y'.
            h (int): Number of neighbors to use before and after the missing value.

        Returns:
            np.ndarray: The array (x or y) with missing values imputed.
                        Missing ones with insufficient context remain None.
        """
        assert target in ("x", "y"), "target must be 'x' or 'y'"
        
        x_filled = np.copy(x).astype(object)
        y_filled = np.copy(y).astype(object)
        
        n = len(x)

        for i in range(n):

            start = max(0, i - h)
            end = min(n, i + h + 1)

            x_window = x[start:end]
            y_window = y[start:end]

            if target == "y" and (y[i] is None or np.isnan(y[i])):

                mask = np.array([
                    y_window[j] is not None and not np.isnan(y_window[j]) and start + j != i
                    for j in range(len(y_window))])

                x_train = np.array(x_window)[mask].reshape(-1, 1)
                y_train = np.array(y_window)[mask]

                if len(x_train) < 2:
                    continue
                try:
                    gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-3, normalize_y=True)
                    gp.fit(x_train, y_train)

                    x_query = np.array([[x[i]]])
                    if np.isnan(x_query).any():
                        continue

                    y_pred = gp.predict(x_query)[0]
                    y_filled[i] = float(y_pred)
                except Exception as e:
                    continue

            elif target == "x" and (x[i] is None or np.isnan(x[i])):

                mask = np.array([
                    x_window[j] is not None and not np.isnan(x_window[j]) and (start + j != i)
                    for j in range(len(x_window)) ])

                y_train = np.array(y_window)[mask].reshape(-1, 1)
                x_train = np.array(x_window)[mask]

                if len(y_train) < 2:
                    continue
                try:
                    gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-3,normalize_y=True)
                    gp.fit(y_train, x_train)

                    y_query = np.array([[y[i]]])
                    if np.isnan(y_query).any():
                        continue

                    x_pred = gp.predict(y_query)[0]
                    x_filled[i] = float(x_pred)
                except Exception as e:
                    continue

        return y_filled if target == "y" else x_filled


# ===================== TEST ======================

if __name__ == "__main__":

    np.random.seed(42)
    x = np.linspace(0, 50, 120).astype(object)
    y = (2 * np.array(x) + np.random.normal(0, 1, 120)).astype(object)

    # Bazı y değerlerini eksiltelim
    for i in [50,60]:
        y[i] = None

    print("Original y with missing:")
    print(y)

    y_filled = Imputator.impute_with_gp(x, y, target="y", h=40)

    print("\nFilled y:")
    print(y_filled)

    # Benzer şekilde x tarafını bozalım
    x_missing = np.copy(x)
    for i in [ 50, 60]:
        x_missing[i] = None

    print("\nOriginal x with missing:")
    print(x_missing)

    x_filled = Imputator.impute_with_gp(x_missing, y_filled, target="x", h=40)

    print("\nFilled x:")
    print(x_filled)