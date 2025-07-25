from pecnet.network import *

class PecnetBuilder:
    """
    This class is a builder for constructing a Pecnet model.

    The PecnetBuilder class provides a fluent interface to sequentially add different components
    to a Pecnet model. It encapsulates the creation logic of the Pecnet and its constituent networks,
    allowing for a more readable and manageable setup.

    Attributes:
        pecnet (Pecnet): An instance of the Pecnet class that this builder will configure and build.

    Methods:
        add_final_network: Adds a final network to the Pecnet.
        add_error_network: Adds an error network to the Pecnet.
        add_variable_network: Adds a variable network to the Pecnet.
        build: Finalizes the construction of the Pecnet and returns the instance.
    """
    def __init__(self):
        self.pecnet = Pecnet()

    def add_variable_network(self, X_train, y_train=None):
        """
        Adds a variable network to the Pecnet model. Supports multiple variable networks.

        The variable network is a customizable part of the model that can be trained on specific
        training data and its different frequency bands. The initial variable network should be provided with both X_train and y_train.
        Subsequent variable networks only require X_train.

        Args:
            X_train: Training input data.
            y_train: Training target data.
        """

        _is_primary_network = (len(self.pecnet.variable_networks) == 0)
        _pre_comp_preds=[]

        if not _is_primary_network:
            _pre_comp_preds = self.pecnet.get_last_compensated_predictions()

        if y_train is None:
            y_train = self.pecnet.get_target_values_for_current_variable_network()

        self.pecnet.add_variable_network(VariableNetwork(X_train,y_train,_is_primary_network,_pre_comp_preds))
        return self

    def add_final_network(self):
        """
        Adds a final network to the Pecnet model.
        The final network is responsible for generating the final predictions based on all
        previous predictions.
        """
        self.pecnet.final_network = FinalNetwork(self.pecnet.get_all_preds())
        return self

    def add_error_network(self):
        """
        Adds an error network to the Pecnet model.
        The error network is used  after a variable network for calculating and compensating the errors in the predictions.
        It automatically gets the shifted compensated errors as input and tries to predict errors of errors
        """
        self.pecnet.error_network = ErrorNetwork(self.pecnet.get_shifted_compensated_errors(),
                                                 self.pecnet.get_last_compensated_predictions())
        return self


    def build(self):
        """
        Returns:
            Pecnet: The fully constructed Pecnet model, with all of its constituent networks.
        """
        return self.pecnet