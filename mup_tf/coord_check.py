"""
Helper functions for performing coord check.
"""
from copy import copy

import pandas as pd
import tensorflow as tf

from mup_tf.layer import extract_layers


def cov(x):
    """Treat `x` as a collection of vectors and its Gram matrix.

    Args:
        x: A tf.tensor that if it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Returns:
        a matrix of size N x N where N is the product of
            the non-last dimensions of `x`.
    """
    if tf.size(x) == 1:
        width = 1
        xx = tf.reshape(x, (1, 1))
    else:
        width = x.shape[-1]
        xx = tf.reshape(x, (-1, width))
    return xx @ tf.transpose(xx) / width


def covoffdiag(x):
    """Get off-diagonal entries of `cov(x)` in a vector.

    Args:
        x: A tf.tensor it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Returns:
        Off-diagonal entries of `cov(x)` in a vector."""
    c = cov(x)
    return c[~tf.eye(c.shape[0], dtype=bool)]


#: dict of provided functions for use in coord check
FDICT = {
    "l1": lambda x: tf.math.reduce_mean(tf.math.abs(x)),
    "l2": lambda x: tf.math.reduce_mean(x**2) ** 0.5,
    "mean": lambda x: tf.math.reduce_mean(x),
    "std": lambda x: tf.math.reduce_std(x),
    "covl1": lambda x: tf.math.reduce_mean(tf.math.abs(cov(x))),
    "covl2": lambda x: tf.math.reduce_mean(cov(x) ** 2) ** 0.5,
    "covoffdiagl1": lambda x: tf.math.reduce_mean(tf.math.abs(covoffdiag(x))),
    "covoffdiagl2": lambda x: tf.math.reduce_mean((covoffdiag(x) ** 2)) ** 0.5,
}


def convert_fdict(d):
    """convert a dict `d` with string values to function values.

    Args:
        d: a dict whose values are either strings or functions
    Returns:
        a new dict, with the same keys as `d`, but the string values are
        converted to functions using `FDICT`.
    """
    return dict(
        [((k, FDICT[v]) if isinstance(v, str) else (k, v)) for k, v in d.items()]
    )


def get_input_output(model, data, t, width, output_fdict=None, skip_tensors=None):
    """
    Args:
        model: a tf.keras.models.Model
        data: a batch of data
        t: int representing the time step
        width: int representing the width of the model
        output_fdict: dict of functions to apply to the output of the model

    Returns:
        a list of dicts where a function in `output_fdict` is applied to the
        output of each layer of the model.
    """
    if output_fdict is None:
        output_fdict = dict(l1=FDICT["l1"])
    else:
        output_fdict = convert_fdict(output_fdict)

    records = []

    output = data
    layers = extract_layers(model, extract_custom=False)

    for i, tensor in enumerate(layers):
        if skip_tensors and skip_tensors in tensor.name:
            continue

        output = tensor(output, training=False)
        name = f"{tensor.name.split('_')[0]}_{i}"
        ret = {"width": width, "module": name, "t": t}

        for fname, f in output_fdict.items():
            ret[fname] = f(output).numpy()

        records.append(ret)

    return records


def _get_coord_data(
    models,
    dataloader,
    optimizer,
    nsteps=3,
    dict_in_out=False,
    dict_out=False,
    flatten_input=False,
    flatten_output=False,
    output_name="loss",
    lossfn="xent",
    filter_module_by_name=None,
    fix_data=True,
    nseeds=1,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
    one_hot_target=False,
    skip_tensors=None,
):
    """Inner method for `get_coord_data`.

    Train the models in `models` with optimizer given by `optimizer` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Args:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a tf.keras.optimizers.Optimizer or an optimizer implemented in `mup_tf`.
        nsteps:
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `tf.reshape(input, (input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `tf.reshape(label, (-1, label.shape[-1]))`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse'] Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given layer names (from
            `model.layers`), or None. If not None, then only modules
            whose name yields True will be recorded.
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Returns:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    """
    df = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        tf.random.set_seed(i)
        for width, model in models.items():
            model = model()
            for t, batch in enumerate(dataloader, 1):
                with tf.GradientTape() as tape:
                    if dict_in_out:
                        outputs = model(**batch, training=True)
                        output = (
                            outputs[output_name]
                            if isinstance(outputs, dict)
                            else outputs[0]
                        )

                    elif dict_out:
                        (data, target) = batch
                        outputs = model(data, training=True)
                        output = (
                            outputs[output_name]
                            if isinstance(outputs, dict)
                            else outputs[0]
                        )

                    else:
                        (data, target) = batch
                        if flatten_input:
                            data = tf.reshape(data, (data.shape[0], -1))
                        output = model(data, training=True)
                        if flatten_output:
                            output = tf.reshape(output, (-1, output.shape[-1]))
                        if one_hot_target:
                            target = tf.one_hot(
                                tf.squeeze(target), depth=output.shape[-1]
                            )

                    if lossfn == "xent":
                        loss = tf.keras.losses.CategoricalCrossentropy(
                            from_logits=True
                        )(target, output)
                    elif lossfn == "mse":
                        loss = tf.keras.losses.MeanSquaredError()(target, output)
                    elif lossfn == "poisson":
                        loss = tf.keras.losses.Poisson()(target, output)
                    else:
                        raise NotImplementedError(f"unknown `lossfn`: {lossfn}")

                    # compute gradients
                    gradients = tape.gradient(loss, model.trainable_variables)

                    df.extend(
                        get_input_output(
                            model, data, t, width, skip_tensors=skip_tensors
                        )
                    )

                    optimizer.apply_gradients(
                        (grad, var)
                        for (grad, var) in zip(gradients, model.trainable_variables)
                        if grad is not None
                    )
                    del gradients

                if t == nsteps:
                    break

            del model

            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)


def get_coord_data(
    models,
    dataloader,
    optimizer="sgd",
    learning_rate=None,
    mup=True,
    filter_trainable_by_name=None,
    skip_tensors=None,
    **kwargs,
):
    """Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Args:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        learning_rate:
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name:
            a function that returns a bool given module names (from
            `model.layers`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps:
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse']. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.layers`), or None. If not None, then only modules
            whose name yields True will be recorded.
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Returns:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    """
    if learning_rate is None:
        learning_rate = 0.1 if optimizer == "sgd" else 1e-3
    if mup:
        from mup_tf import MuAdam as Adam
        from mup_tf import MuSGD as SGD
    else:
        from tensorflow.keras.optimizers import SGD, Adam

    def get_trainable(model):
        params = model.trainable_weights
        if filter_trainable_by_name is not None:
            params = []
            for tensor in model.trainable_weights:
                if filter_trainable_by_name(tensor.name):
                    params.append(tensor)
        return params

    if optimizer == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer is None:
        raise ValueError("optimizer should be sgd|adam")

    data = _get_coord_data(
        models, dataloader, optimizer, skip_tensors=skip_tensors, **kwargs
    )
    data["optimizer"] = optimizer
    data["learning_rate"] = learning_rate
    return data


def plot_coord_data(
    df,
    y="l1",
    save_to=None,
    suptitle=None,
    x="width",
    hue="module",
    legend="auto",
    name_contains=None,
    name_not_contains=None,
    loglog=True,
    logbase=2,
    face_color=None,
    subplot_width=20,
    subplot_height=16,
):
    """Plot coord check data `df` obtained from `get_coord_data`.

    Args:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'module'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains:
            only plot modules whose name contains `name_contains`
        name_not_contains:
            only plot modules whose name does not contain `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
        subplot_width, subplot_height:
            The width and height for each timestep's subplot. More precisely,
            the figure size will be
                `(subplot_width*number_of_time_steps, subplot_height)`.
            Default: 5, 4

    Returns:
        the `matplotlib` figure object
    """
    ### preprocessing
    df = copy(df)
    df = df[df.module != ""]

    if name_contains is not None:
        df = df[df["module"].str.contains(name_contains)]
    elif name_not_contains is not None:
        df = df[~(df["module"].str.contains(name_not_contains))]

    ts = df.t.unique()

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    ### plot
    fig = plt.figure(figsize=(subplot_width * len(ts), subplot_height))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    for t in ts:
        plt.subplot(1, len(ts), t)
        sns.lineplot(
            x=x, y=y, data=df[df.t == t], hue=hue, legend=legend if t == 1 else None
        )
        plt.title(f"t={t}")
        if t != 1:
            plt.ylabel("")
        if loglog:
            plt.loglog(base=logbase)
    if suptitle:
        plt.suptitle(suptitle)
    if save_to is not None:
        plt.savefig(save_to)
        print(f"coord check plot saved to {save_to}")

    return fig
