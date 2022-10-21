import tensorflow as tf


class MuOutput(tf.keras.layers.Layer):
    """Prototype for all output linear layers.
    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).
    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    """

    def __init__(
        self,
        *args,
        readout_zero_init=False,
        output_mult=1.0,
        kernel_initializer="he_uniform",
        bias_initializer="random_normal",
        **kwargs
    ):
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        if self.readout_zero_init:
            kernel_initializer = tf.keras.initializers.Constant(0)
            bias_initializer = tf.keras.initializers.Constant(0)

        super().__init__(
            *args,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs
        )

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.kernel.assign(tf.zeros_like(self.kernel))
            if self.bias is not None:
                self.bias.assign(tf.zeros_like(self.bias))
        else:
            if isinstance(self.kernel.kernel_initializer, str):
                self.kernel.kernel_initializer = tf.keras.initializers.get(
                    self.kernel.kernel_initializer
                )
            if isinstance(self.bias.bias_initializer, str):
                self.bias.bias_initializer = tf.keras.initializers.get(
                    self.bias.bias_initializer
                )

            self.kernel.assign(
                self.kernel.kernel_initializer(self.kernel.shape, self.kernel.dtype)
            )
            self.bias.assign(
                self.bias.bias_initializer(self.bias.shape, self.bias.dtype)
            )

    def width_mult(self):
        """Calculates the width multiplier for this layer.

        NOTE: We add a hack to return 1 if the layer has not been built yet (or just built)
        since `.build()` is called before `.set_base_shapes()`. To create variables
        in Tensorflow, you must call `.build()` first, otherwise there will no be `.kernel`
        attributed.  Internally, `build` calls `.call()` which calls `.width_mult()`.
        So we need to make sure that the model can be built before calling `.set_base_shapes()`.

        Returns:
            The width multiplier for this layer.
        """
        if hasattr(self.kernel, "infshape"):
            return self.kernel.infshape.width_mult()
        elif hasattr(self, "_has_rescaled_params"):
            raise ValueError("No infshape found for this layer.")
        else:
            # TODO this is a hack so when a layer is built, it can still call
            # Should we raise a warning? It's not easy to check if a layer was
            # just built or not.
            return 1

    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.assign(self.bias * self.width_mult() ** 0.5)
        self.kernel.assign(self.kernel * self.width_mult() ** 0.5)
        self._has_rescaled_params = True

    def call(self, x):
        output = super().call(self.output_mult * x / self.width_mult())
        return output


class MuReadout(MuOutput, tf.keras.layers.Dense):
    """Drop-in replacement for all output Linear layers."""

    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, **kwargs):
        super().__init__(
            *args,
            readout_zero_init=readout_zero_init,
            output_mult=output_mult,
            **kwargs
        )


class MuSharedReadout(MuReadout):
    """`MuReadout` with weights shared with an `tf.keras.layers.Embedding` layer."""

    def __init__(self, weight, bias=True, **kwargs):
        """Args:
        weight: should be weight of an `tf.keras.layers.Embedding` layer
        """
        super().__init__(*weight.shape, bias=bias, **kwargs)
        self.weight = weight


class MuOutConv1D(MuOutput, tf.keras.layers.Conv1D):
    """Drop-in replacement for all output Conv1d layers."""

    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, **kwargs):
        super().__init__(
            *args,
            readout_zero_init=readout_zero_init,
            output_mult=output_mult,
            **kwargs
        )


def rescale_bias(linear):
    """Rescale bias in tf.keras.layers.Dense or tf.keras.layers.ConvND layers to convert SP initialization to μP initialization.
    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.

    Args:
        linear: a tf.keras.layers.Dense or tf.keras.layers.ConvND layer
    """
    if hasattr(linear, "_has_rescaled_params") and linear._has_rescaled_params:
        raise RuntimeError(
            "`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n"
            "If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n"
            "To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call."
        )
    if linear.bias is None:
        return
    fanin_mult = linear.trainable_weights[0].infshape[1].width_mult()
    linear.bias = linear.bias * fanin_mult**0.5
    linear._has_rescaled_params = True


def reinit_layer(
    layer, kernel_initializer="he_uniform", bias_initializer="random_normal"
):
    """Reinitializes layers to have μP variance

    Args:
        layer: a tf.keras.layers.Dense or tf.keras.layers.ConvND layer
        kernel_initializer: initializer for kernel (str or tf.keras.initializers.Initializer)
        bias_initializer: initializer for bias (str or tf.keras.initializers.Initializer)
    """
    if (
        layer.kernel_initializer == kernel_initializer
        and layer.bias_initializer == bias_initializer
    ):
        return layer

    if isinstance(kernel_initializer, str):
        kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    if isinstance(bias_initializer, str):
        bias_initializer = tf.keras.initializers.get(bias_initializer)

    layer.kernel_initializer = kernel_initializer
    layer.bias_initializer = bias_initializer

    if layer.built:
        # Reset weights and biases if layer is already built
        layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape, layer.dtype))
        if layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape, layer.dtype))

    return layer


def extract_layers(seq_layers, extract_custom=True):
    """Extract layers from a sequential layer.

    Args:
        seq_layers: a tf.keras.layers.Sequential

    Returns:
        a list of layers from the sequential layer.
    """
    layers = []
    for layer in seq_layers.layers:
        if isinstance(layer, tf.keras.Sequential):
            layers.extend(extract_layers(layer, extract_custom=extract_custom))
        # custom layers track their sublayers in `obj.layers`
        elif extract_custom and hasattr(layer, "layers"):
            layers.extend(extract_layers(layer, extract_custom=extract_custom))
        elif not layer.trainable or isinstance(
            layer, (tf.keras.layers.Activation, tf.keras.layers.Dropout)
        ):
            continue
        else:
            layers.append(layer)
    return layers
