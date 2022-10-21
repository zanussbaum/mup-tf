import re
from copy import deepcopy

import tensorflow as tf
import yaml

from mup_tf.infshape import InfShape, zip_infshape
from mup_tf.layer import MuReadout, extract_layers, reinit_layer

__BSH_COMMENT__ = """\
# This is a base shape file encoded in yaml
# - `null` indicates a dimension is "finite", i.e. a non-"width" dimension
# - a number indicates the base dimension of an "infinite" dimension, i.e. some notion of "width"
"""


def get_base_name(name):
    """Get the base name of a layer. e.g. returns dense from dense

    Args:
        name: name of a layer

    Returns:
        base name of a layer
    """
    return re.sub(r"_[\d]+", "", name)


def count_layers(shapes):
    """Count the number of layers in a dict of shapes.

    Args:
        shapes: an iterable of shapes

    Returns:
        dict of base layer name to number of layers
    """
    counter = {}
    for name in shapes:
        split = get_base_name(name)
        count = counter.get(split, 0)
        counter[split] = count + 1

    return counter


def rename_tensor_names(shapes, tensors):
    """Rename the tensor names in a dict of shapes to be locally named.

    Tensorflow names tensors with a global name. For example, imagine you have
    two models each with a dense layer. The first dense layer in the first model
    would be named `dense/kernel:0` and the first dense layer in the second model
    woudl be named `dense_1/kernel:0`. This function renames the tensors to be
    locally named so each tensor is named `dense/kernel:0` in both models. This
    is needed so that we can map layers across models.

    Args:
        shapes: a dict of shapes where the keys are tensor names and the values
            are the shapes of the tensors
        tensors: a list of tensors.

    Returns:
        a dict of shapes where the keys are locally named tensor names

    """
    layer_name_count = count_layers(shapes)
    counted_tensors = {key: 0 for key in layer_name_count}

    renamed_tensors = {}
    for tensor in tensors:
        base_name = get_base_name(tensor.name)
        shape = shapes[tensor.name]
        if base_name not in counted_tensors:
            counted_tensors[base_name] = 0
        else:
            counted_tensors[base_name] += 1
        new_name = f"{base_name}_layer_{counted_tensors[base_name]}"
        renamed_tensors[new_name] = shape

        tensor.mapped_name = new_name

    assert len(renamed_tensors) == len(tensors)
    assert list(renamed_tensors.values()) == list(shapes.values())

    return renamed_tensors


def get_shapes(model):
    """Gets the shapes of each weight in a model.

    Args:
        model: a `tf.keras.Model`

    Returns:
        a dict of shapes where the keys are the names of the weights and the values
        are the shapes of the weights
    """
    tensor2shape = {tensor.name: tensor.shape for tensor in model.trainable_weights}

    tensor2shape = rename_tensor_names(tensor2shape, model.trainable_weights)
    return tensor2shape


def get_infshapes(model):
    """Gets the `mup_tf.InfShape` of each weight in a model.

    Args:
        model: a `tf.keras.Model`

    Returns:
        a dict of shapes where the keys are the names of the weights and the values
        are the `mup_tf.InfShape` of the weights
    """
    tensor_names = [tensor.name for tensor in model.trainable_weights]
    tensor2infshape = {
        tensor.name: tensor.infshape for tensor in model.trainable_weights
    }
    tensor2infshape = rename_tensor_names(tensor2infshape, tensor_names)
    return tensor2infshape


def save_base_shapes(model_or_shapes, file):
    """Saves the shapes of a model to a file.

    Args:
        model_or_shapes: a `tf.keras.Model` or a dict of shapes
        file: a file to save the shapes to

    Raises:
        ValueError: if `model_or_shapes` is not a `tf.keras.Model` or a dict of shapes
    """
    if isinstance(model_or_shapes, tf.keras.layers.Layer):
        sh = get_infshapes(model_or_shapes)
    elif isinstance(model_or_shapes, dict):
        sh = deepcopy(model_or_shapes)
    else:
        raise ValueError()
    sh = {k: s.base_shape() for k, s in sh.items()}
    s = yaml.dump(sh, None, indent=4)
    s = __BSH_COMMENT__ + s
    with open(file, "w") as f:
        f.write(s)


def load_base_shapes(filename):
    """Get a dict of `InfShape` from a filename.

    Args:
        filename: a filename to load the shapes from

    Returns:
        a dict of `InfShape` where the keys are the names of the weights and the values
    """
    with open(filename, "r") as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}


def _extract_shapes(x):
    """
    Args:
        x: can be any of the following:
            - `tf.keras.layers.Layer`
            - dict of shapes
            - dict of `InfShape`
            - str of path to a base shapes (.bsh) file
    Returns:
        If `x` is dict of `InfShape`, then output itself.
        If `x` is path, then output a dict of `InfShapes` loaded from `x`.
        Else, output the shapes (not `InfShape`) associated to `x`
    """
    if isinstance(x, tf.keras.layers.Layer):
        x_shapes = get_shapes(x)
    elif isinstance(x, dict):
        x_shapes = deepcopy(x)
    elif isinstance(x, str):
        # x is file name
        x_shapes = load_base_shapes(x)
    else:
        raise ValueError(f"unhandled x type: {type(x)}")
    return x_shapes


def assert_equal_shapes(base_shapes, shapes):
    """Assert that two dicts of shapes are equal.

    Args:
        base_shapes: a dict of shapes
        shapes: a dict of shapes
    """
    base_layers_counter = count_layers(base_shapes)

    layers_counter = count_layers(shapes)

    if layers_counter != base_layers_counter:
        raise ValueError(
            f"layers_counter != base_layers_counter: {layers_counter} != {base_layers_counter}"
        )


def _zip_infshape_dict(base_shapes, shapes):
    """Make a dict of `InfShape` from two dicts of shapes.

    Args:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Returns:
        dict of `InfShape` using `zip_infshape`
    """
    assert_equal_shapes(base_shapes, shapes)

    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes


def zip_infshapes(base, target):
    """Make a dict of `InfShape` from models or dicts.

    Args:
        base: a base `tf.keras.layers.Layer` or a dict of shapes
        target: a target `tf.keras.layers.Layer` or a dict of shapes
    Returns:
        dict of `InfShape` using `zip_infshape`
    """
    base_shapes = _extract_shapes(base)
    target_shapes = _extract_shapes(target)
    return _zip_infshape_dict(base_shapes, target_shapes)


def clear_dims(infshape_dict):
    """
    Args:
        infshape_dict: dict of `InfShape`
    Returns:
        the same dict but where all `InfDim` in all `InfShape`
        have their `dim` attribute set to None
    """
    d = deepcopy(infshape_dict)
    for _, v in d.items():
        for infdim in v:
            infdim.dim = None
    return d


def make_base_shapes(base_shapes, delta_shapes, savefile=None):
    """Make a base shape object from a base model/shapes and a delta model/shapes.

    Args:
        base:
            a base `tf.keras.models.Model` or a dict of shapes
        delta:
            a "delta" model or a dict of shapes, for the sole purpose of
            determining which dimensions are "width" and will be scaled up and
            down in the target model.
        savefile:
            if a string, then the resulting base shape object is serialized to
            this location via yaml encoding.
    Returns:
        base infshapes
    """
    bsh = clear_dims(zip_infshapes(base_shapes, delta_shapes))
    if savefile is not None:
        save_base_shapes(bsh, savefile)
    return bsh


def apply_infshapes(model, infshapes):
    """Apply a dict of `InfShape` to a model.

    Args:
        model: a `tf.keras.models.Model`
        infshapes: a dict of tensor_name: `InfShape`
    """
    for tensor in model.trainable_weights:
        tensor.infshape = infshapes[tensor.mapped_name]


def set_base_shapes(
    model, base, rescale_params=True, delta=None, savefile=None, do_assert=True
):
    """Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Args:
        model: tf.keras.models.Model instance
        base: The base model.
            Can be tf.keras.models.Model, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        rescale_params:
            assuming the model is initialized using the default init (or
            He initialization etc that scale the same way with fanin): If True
            (default), rescales parameters to have the correct (μP) variances.
        do_assert:
    Returns:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    """
    if not model.built:
        raise ValueError("model must be built before setting base shapes")

    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)
    if rescale_params:
        layers = extract_layers(model)
        for layer in layers:
            if isinstance(layer, MuReadout):
                layer._rescale_parameters()
            elif isinstance(
                layer,
                (tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv1D),
            ):
                reinit_layer(layer)

    name2shape = {k.name: infshapes[k.mapped_name] for k in model.trainable_weights}
    return name2shape


def assert_hidden_size_inf(model):
    """
    This tests for any `Dense` whose output dimension is finite but input
    dimension is infinite and is not of type `MuReadout`. Such `Dense`
    modules should not exist in a correctly parametrized models.

    Args:
        model: a `tf.keras.models.Model`
    """
    for tensor in model.trainable_weights:
        if isinstance(tensor, tf.keras.layers.Dense) and not isinstance(
            tensor, MuReadout
        ):
            if not tensor.infshape[0].isinf() and tensor.infshape[1].isinf():
                raise ValueError(
                    f"{tensor.mapped_name} has infinite fan-in and finite fan-out dimensions but is not type `MuReadout`. "
                    "To resolve this, either change the module to `MuReadout` or change the fan-out to an infinite dimension."
                )
