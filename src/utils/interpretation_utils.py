from models.nlp.transformer import transformer
from models.nlp.fnn import fnn
from models.nlp import cnn1d, rnn, lstm
from utils import nn_utils
import joblib

def get_loaded_model(model_file_path, model_config, model_type):
    model = None
    if model_type == "lr" or model_type == "rf":
        model = joblib.load(model_file_path)
        print(model)
        return model
    if model_type == "transformer":
        model = transformer.get_transformer_model(model_config)
    elif model_type == "fnn":
        model = fnn.get_fnn_model(model_config)
    elif model_type == "cnn":
        model = cnn1d.get_cnn_model(model_config)
    elif model_type == "rnn":
        model = rnn.get_rnn_model(model_config)
    elif model_type == "lstm":
        model = lstm.get_lstm_model(model_config)
    else:
        print(f"ERROR: Unsupported model type - '{model_type}'.\n" +
              f"Supported values 'lr', 'rf', 'fnn', 'cnn', 'rnn', 'lstm', and 'transformer'.")
        return

    model.load_state_dict(torch.load(model_file_path))
    model = nlp_model.to(nn_utils.get_device())
    return model