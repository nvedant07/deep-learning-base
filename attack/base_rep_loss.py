## Base class for any loss defined in the representation space

import torch as ch

class BaseLoss:

    def __init__(self, lpnorm_type):
        self.lpnorm_type = lpnorm_type

    def set_target_inps(self, target_inp) -> None:
        ## implement as necessary in derived classes
        pass

    def __call__(self, model1, model2, inp, targ1, targ2) -> ch.Tensor:
        raise NotImplementedError('Call must be implemented in inherited class')
