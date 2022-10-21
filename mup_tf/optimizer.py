import tensorflow as tf


class MuAdam(tf.keras.optimizers.Adam):
    """Adam optimizer with μP-aware learning rate scaling.

    This behaves like the tf.keras.optimizers.Adam optimizer, but will scale
    the learning rate by the width multiplier.
    """

    def __init__(self, *args, **kwargs):
        self.infshapes = kwargs.pop("infshapes", None)
        super().__init__(*args, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # changed from https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/adam.py#L198
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        if hasattr(var, "infshape"):
            infshape = var.infshape
        else:
            infshape = self.infshapes[var.name]

        ninf = infshape.ninf()
        lr = coefficients["lr_t"]

        # from table 8 in https://arxiv.org/pdf/2203.03466.pdf
        if ninf == 2:
            lr /= infshape.width_mult()

        if not self.amsgrad:
            return tf.raw_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=coefficients["beta_1_power"],
                beta2_power=coefficients["beta_2_power"],
                lr=lr,
                beta1=coefficients["beta_1_t"],
                beta2=coefficients["beta_2_t"],
                epsilon=coefficients["epsilon"],
                grad=grad,
                use_locking=self._use_locking,
            )
        else:
            vhat = self.get_slot(var, "vhat")
            return tf.raw_ops.ResourceApplyAdamWithAmsgrad(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                vhat=vhat.handle,
                beta1_power=coefficients["beta_1_power"],
                beta2_power=coefficients["beta_2_power"],
                lr=lr,
                beta1=coefficients["beta_1_t"],
                beta2=coefficients["beta_2_t"],
                epsilon=coefficients["epsilon"],
                grad=grad,
                use_locking=self._use_locking,
            )


class MuSGD(tf.keras.optimizers.SGD):
    """Adam optimizer with μP-aware learning rate scaling.

    This behaves like the tf.keras.optimizers.SGD optimizer, but will scale
    the learning rate by the width multiplier.
    """

    def __init__(self, *args, **kwargs):
        self.infshapes = kwargs.pop("infshapes", None)
        super().__init__(*args, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # changed from https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/gradient_descent.py#L132
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        if hasattr(var, "infshape"):
            infshape = var.infshape
        else:
            infshape = self.infshapes[var.name]

        ninf = infshape.ninf()
        lr = coefficients["lr_t"]

        # from table 8 in https://arxiv.org/pdf/2203.03466.pdf
        if ninf == 1:
            lr *= infshape.width_mult()
        if ninf == 2:
            lr /= infshape.fanin_fanout_mult_ratio()

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return tf.raw_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=momentum_var.handle,
                lr=lr,
                grad=grad,
                momentum=coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov,
            )
        else:
            return tf.raw_ops.ResourceApplyGradientDescent(
                var=var.handle, alpha=lr, delta=grad, use_locking=self._use_locking
            )
