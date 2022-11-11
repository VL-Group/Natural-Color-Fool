import torch
from torch_attack.ensemble import get_ensemble_models


class BaseAttack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name:str, models:dict, config):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (dict): model to attack, value is torch.nn.Module
            return_type (str): 'float' or 'int'. (Default: 'float')
        """
        self.attack_name = name
        self.models = models
        self.models_name = list(models.keys())
        self.model_mode = config.model_mode
        self.device = next(models[self.models_name[0]].parameters()).device

        self._attack_mode = config['attack_mode'] # 'untargeted' or 'targeted(least-likely)' or 'targeted(random)' eta
        self._return_type = config['return_type']  # 'float' or 'int'
        self._supported_mode = ['untargeted']  
        self._ensemble = False # True or False

        self.set_ensemble_mode()
        if self._attack_mode != 'untargeted':
            self.set_mode_targeted_by_function()

        # self._model_training = False # True or False
        # self._batchnorm_training = False # True or False
        # self._dropout_training = False # True or False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_ensemble_mode(self, ensemble_model_function=get_ensemble_models):
        r"""
        Set ensemble method of attack models.
        
        """
        if len(self.models_name) > 1:
            self._ensemble_model_function = ensemble_model_function
            self._ensemble = True
            print("Attack multiple model")
        else:
            self._ensemble = False
            print("Attack a single models")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def _get_output(self, images):
        r"""
        Function for changing the ensemblemode.
        Return models output.
        """        
        if self._ensemble:
            return self._ensemble_model_function(self.models, self.models_name, self.model_mode, images)
        else:
            return self.models[self.models_name[0]](images)


    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function:
            return self._target_map_function(images, labels)
        raise ValueError('Please define target_map_function.')

    def __call__(self, *input, **kwargs):

        for name in self.models_name:
            self.models[name].eval()

        images = self.forward(*input, **kwargs)

        if self._return_type == 'int':
            images = (images*255).type(torch.uint8)

        return images
