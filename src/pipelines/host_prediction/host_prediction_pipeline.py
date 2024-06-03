from pipelines.host_prediction import host_prediction_nlp_models_pipeline, host_prediction_baseline_models_pipeline


def execute(config):
    # input_settings
    input_settings = config["input_settings"]

    # output settings
    output_settings = config["output_settings"]

    # classification settings
    classification_settings = config["classification_settings"]
    type = classification_settings["model_type"]

    if type == "baseline":
        host_prediction_baseline_models_pipeline.execute(input_settings, output_settings, classification_settings)
    elif type == "nlp":
        host_prediction_nlp_models_pipeline.execute(input_settings, output_settings, classification_settings)