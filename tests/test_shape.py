import tensorflow as tf

from mup_tf import set_base_shapes


def test_set_infshape():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(1),
        ]
    )

    model.build(input_shape=(None, 100, 4))

    base_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(8), tf.keras.layers.Dense(8), tf.keras.layers.Dense(1)]
    )

    base_model.build(input_shape=(None, 100, 4))

    set_base_shapes(model, base_model)

    assert all(t.infshape is not None for t in model.trainable_weights)
