import argparse
import os

import numpy as np
import tensorflow as tf

from mup_tf import MuReadout, get_coord_data, plot_coord_data, set_base_shapes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--mup", action="store_true")
    parser.add_argument("--nseeds", type=int, default=1)
    parser.add_argument("--nsteps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)

    return parser.parse_args()


def create_model(width, output_units, mup):
    if mup:
        last_layer = MuReadout(output_units, readout_zero_init=True)
    else:
        last_layer = tf.keras.layers.Dense(output_units)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(width, activation="relu"),
            tf.keras.layers.Dense(width, activation="relu"),
            last_layer,
        ]
    )

    return model


def create_mlp(width, base_width, output_units, mup=False, input_shape=None):
    def gen(width):
        def f():
            model = create_model(width, output_units, mup=mup)
            model.build(input_shape)
            if not mup:
                set_base_shapes(model, None)
                return model

            base_model = create_model(base_width, output_units, mup=mup)
            base_model.build(input_shape)
            set_base_shapes(model, base_model)

            return model

        return f

    return gen(width)


def lazy_model(widths, base_width, output_units, mup=False, input_shape=None):
    return {
        w: create_mlp(
            width=w,
            base_width=base_width,
            output_units=output_units,
            mup=mup,
            input_shape=input_shape,
        )
        for w in widths
    }


args = parse_args()
mup = args.mup
optimizer = args.optimizer
nseeds = args.nseeds
nsteps = args.nsteps
lr = args.lr

widths = 2 ** np.arange(7, 15)
models = lazy_model(
    widths=widths,
    base_width=128,
    output_units=10,
    mup=mup,
    input_shape=(None, 32 * 32 * 3),
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train_norm = (x_train / 127.5) - 1


train_loader = (
    tf.data.Dataset.from_tensor_slices((x_train_norm, y_train)).shuffle(10000).batch(2)
)

df = get_coord_data(
    models,
    train_loader,
    mup=mup,
    learning_rate=lr,
    optimizer=optimizer,
    flatten_input=True,
    nseeds=nseeds,
    nsteps=nsteps,
    one_hot_target=True,
)

prm = "μP" if mup else "SP"

if lr is None:
    lr = 0.1 if optimizer == "sgd" else 1e-3

if not os.path.exists("plots"):
    os.mkdir("plots")

plot_coord_data(
    df,
    legend="full",
    save_to=f"plots/{prm.lower()}_MLP_{optimizer}_lr{lr}_nseeds{nseeds}_coord.png",
    suptitle=f"{prm} MLP {optimizer} lr={lr} nseeds={nseeds}",
    face_color="xkcd:light grey" if not mup else None,
)
