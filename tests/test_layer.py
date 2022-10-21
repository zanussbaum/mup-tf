import tensorflow as tf

from mup_tf import InfDim, reinit_layer, rescale_bias


def test_reinit():
    dense = tf.keras.layers.Dense(64)

    reinit = reinit_layer(dense)

    assert isinstance(reinit.kernel_initializer, tf.keras.initializers.HeUniform)
    assert isinstance(reinit.bias_initializer, tf.keras.initializers.RandomNormal)

    dense.build(input_shape=(None, 32))

    reinit = reinit_layer(dense, kernel_initializer="ones", bias_initializer="zeros")

    assert isinstance(reinit.kernel_initializer, tf.keras.initializers.Ones)
    assert isinstance(reinit.bias_initializer, tf.keras.initializers.Zeros)
    assert tf.math.reduce_all(tf.equal(reinit.kernel, tf.ones([32, 64])))
    assert tf.math.reduce_all(tf.equal(reinit.bias, tf.zeros([64])))


def test_rescale():
    dense = tf.keras.layers.Dense(64, bias_initializer="ones")

    dense.build(input_shape=(None, 32))

    infdim = InfDim(1, 4)
    dense.kernel.infshape = [None, infdim]

    rescale_bias(dense)
    assert hasattr(dense, "_has_rescaled_params")
    assert tf.math.reduce_all(tf.math.equal(dense.bias, 2 * tf.ones([64])))
