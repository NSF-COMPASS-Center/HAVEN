from prediction import baseline_models_prediction, nlp_models_prediction, perturbed_dataset_prediction


def execute(config):
    # input_settings
    input_settings = config["input_settings"]

    # output settings
    output_settings = config["output_settings"]

    # classification settings
    classification_settings = config["classification_settings"]
    type = classification_settings["model_type"]

    if type == "baseline":
        baseline_models_prediction.execute(input_settings, output_settings, classification_settings)
    elif type == "nlp":
        nlp_models_prediction.execute(input_settings, output_settings, classification_settings)
    elif type == "perturbed_dataset_prediction":
        perturbed_dataset_prediction.execute(input_settings, output_settings, classification_settings)