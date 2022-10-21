from pkg_resources import get_distribution

from mup_tf.coord_check import get_coord_data, plot_coord_data
from mup_tf.infshape import InfDim, InfShape, zip_infshape
from mup_tf.layer import (
    MuOutConv1D,
    MuOutput,
    MuReadout,
    MuSharedReadout,
    reinit_layer,
    rescale_bias,
)
from mup_tf.optimizer import MuAdam, MuSGD
from mup_tf.shape import make_base_shapes, set_base_shapes

__version__ = get_distribution("mup_tf").version
__author__ = "Zach Nussbaum"

__all__ = [
    "InfShape",
    "InfDim",
    "zip_infshape",
    "MuAdam",
    "MuOutput",
    "MuReadout",
    "MuSGD",
    "MuSharedReadout",
    "MuOutConv1D",
    "rescale_bias",
    "reinit_layer",
    "set_base_shapes",
    "make_base_shapes",
    "get_coord_data",
    "plot_coord_data",
]
