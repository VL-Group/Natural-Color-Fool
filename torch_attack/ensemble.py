
# The default integrated attack method used.
def get_ensemble_models(models, models_names, model_mode, images):
    r"""
    Function for ensemble models
    """
    logit, aux_logit = 0.0, 0.0
    for model_name in models_names:
        if model_mode == 'torch':
            logit += models[model_name](images)
        elif model_mode == 'tf':
            output = models[model_name](images)
            logit += output[0]
            if len(output) >= 2:
                aux_logit += output[1]
    logit /= len(models_names)
    if model_mode == 'tf' and len(output) >= 2:
        aux_logit /= len(models_names)
        return [logit, aux_logit]
    return logit

