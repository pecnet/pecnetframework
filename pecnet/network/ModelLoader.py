from pecnet.models import BasicNN

def train_or_load_model(x, y, mode, model_list=None, model_index=0):
    """
    Trains or loads a Machine Learning model and returns the predictions.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Target data.
        mode (str): Mode of operation - 'train' or 'test'.
        model_list (list, optional): List to store or retrieve models.
        model_index (int, optional): Index of the model to load in test mode.

    Returns:
        model (BasicNN, etc.): The trained or loaded model.
        preds (np.ndarray): Predictions made by the model.
    """

    sample_size = x.shape[0]
    input_sequence_size = x.shape[1]
    output_sequence_size = y.shape[1]

    if mode == 'train':
        model = BasicNN(sample_size, input_sequence_size, output_sequence_size)
        model.fit(x, y)
        if model_list is not None:
            model_list.append(model)
    elif mode == 'test':
        model = model_list[model_index]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    preds = model.predict(x)
    return model, preds
