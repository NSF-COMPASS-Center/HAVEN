from pipelines.virus_host_prediction_baseline import host_prediction_std_ml_models_pipeline, host_prediction_dl_models_pipeline


def execute(config):
    # input_settings
    input_settings = config["input_settings"]

    # output settings
    output_settings = config["output_settings"]

    # classification settings
    classification_settings = config["classification_settings"]
    type = classification_settings["model_type"]

    if type == "std_ml": # standard machine learning models
        host_prediction_baseline_models_pipeline.execute(input_settings, output_settings, classification_settings)
    elif type == "dl": # deep learning models
        host_prediction_nlp_models_pipeline.execute(input_settings, output_settings, classification_settings)