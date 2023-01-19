import tensorflow as tf


class MuAdam(tf.keras.optimizers.experimental.Adam):
    """Adam optimizer with μP-aware learning rate scaling.

    This behaves like the tf.keras.optimizers.Adam optimizer, but will scale
    the learning rate by the width multiplier.
    """

    def __init__(self, *args, **kwargs):
        self.infshapes = kwargs.pop("infshapes", {})
        super().__init__(*args, **kwargs)

    def update_step(self, gradient, variable):
        # changed from https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/optimizers/optimizer_experimental/adam.py#L143
        # TF 2.10 > uses optimizer_experimental
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)

        ##### Here's the only change ####
        if hasattr(variable, "infshape"):
            infshape = variable.infshape
        else:
            infshape = self.infshapes.get(variable.name, None)

        if infshape:
            ninf = infshape.ninf()
        else:
            ninf = None

        if ninf and ninf == 2:
            lr = tf.Variable(lr / infshape.width_mult(), trainable=False)

        #######################################

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(gradient.values * (1 - self.beta_1), gradient.indices)
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2), gradient.indices
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))


class MuSGD(tf.keras.optimizers.experimental.SGD):
    """Adam optimizer with μP-aware learning rate scaling.

    This behaves like the tf.keras.optimizers.SGD optimizer, but will scale
    the learning rate by the width multiplier.
    """

    def __init__(self, *args, **kwargs):
        self.infshapes = kwargs.pop("infshapes", None)
        super().__init__(*args, **kwargs)

    def update_step(self, gradient, variable):
        # changed from https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/optimizers/optimizer_experimental/sgd.py#L143
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        ##### Here's the only change ####
        if hasattr(variable, "infshape"):
            infshape = variable.infshape
        else:
            infshape = self.infshapes.get(variable.name, None)

        if infshape:
            ninf = infshape.ninf()
        else:
            ninf = None

        # from table 8 in https://arxiv.org/pdf/2203.03466.pdf
        if ninf and ninf == 1:
            lr = tf.Variable(lr * infshape.width_mult(), trainable=False)
        if ninf and ninf == 2:
            lr = tf.Variable(lr / infshape.fanin_fanout_mult_ratio(), trainable=False)
        #######################################

        m = None
        var_key = self._var_key(variable)
        if self.momentum != 0:
            momentum = tf.cast(self.momentum, variable.dtype)
            m = self.momentums[self._index_dict[var_key]]

        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(-gradient.values * lr, gradient.indices)
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Dense gradients
            if m is not None:
                m.assign(-gradient * lr + m * momentum)
                if self.nesterov:
                    variable.assign_add(-gradient * lr + m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-gradient * lr)
