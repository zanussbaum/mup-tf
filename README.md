# MUP for Tensorflow

This is a Tensorflow 2 (very preliminary) port of Yang and Hu et al.'s [μP repo](https://github.com/microsoft/mup)

## Installation

To install, you can either clone the repo and install the package locally, or install it from pyPI.

```bash
pip install mup-tf
```

## Install from Source

```bash
git clone https://github.com/zanussbaum/mup-tf.git
pip install -e .
```

## Basic Usage

This has been adapted from the original MuP repo.

```python
import tensorflow as tf
from mup_tf import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam

class MyModel(tf.keras.Model):
    def __init__(self, width, ...):
        ...
        ### In model definition, replace output layer with MuReadout
        # readout = tf.keras.layers.Dense(d_out)
        readout = MuReadout(d_out)
        ### If tying weights with an input Embedding layer, do
        # readout = MuSharedReadout(input_layer.weight)
        ...
    def call(self, ...):
        ...
        ### If using a transformer, make sure to use
        ###   1/d instead of 1/sqrt(d) attention scaling
        # attention_scores = query @ key.T / d**0.5
        attention_scores = query @ key.T * 8 / d
        ### We use 8/d instead of 1/d here to be backward compatible
        ###   with 1/d**0.5 when d=64, a common head dimension.
        ...

### Instantiate a base model
base_model = MyModel(width=1)
### Instantiate a "delta" model that differs from the base model
###   in all dimensions ("widths") that one wishes to scale.
### Here it's simple, but e.g., in a Transformer, you may want to scale
###   both nhead and dhead, so the delta model should differ in both.
delta_model = MyModel(width=2) 

### Instantiate the target model (the model you actually want to train).
### This should be the same as the base model except 
###   the widths could be potentially different.
### In particular, base_model and model should have the same depth.
model = MyModel(width=100)

### Set base shapes
### When `model` has same parameter shapes as `base_model`,
###   `model` behaves exactly the same as `base_model`
###   (which is in Tensorflow's default parametrization).
###   This provides backward compatibility at this particular model size.
###   Otherwise, `model`'s init and LR are scaled by μP.
### IMPORTANT: this should be called as soon as possible,
###   before re-initialization and optimizer definition.
infshapes = set_base_shapes(model, base_model, delta=delta_model)

### Alternatively, one can save the base model shapes in a file
# make_base_shapes(base_model, delta_model, filename)
### and later set base shapes directly from the filename
# set_base_shapes(model, filename)
### This is useful when one cannot fit both 
###   base_model and model in memory at the same time

### Replace your custom init, if any
for param in model.parameters():
    ### If initializing manually with fixed std or bounds,
    ### then replace with same function from mup.init
    # torch.nn.init.uniform_(param, -0.1, 0.1)
    mup.init.uniform_(param, -0.1, 0.1)
    ### Likewise, if using
    ###   `xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_`
    ### from `torch.nn.init`, replace with the same functions from `mup.init`

### Use the optimizers from `mup.optim` instead of `tf.keras.optimizers`
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
opt_kwargs = {"infshapes": name2shapes}
# need to pass in infshapes to optimizer if you are using tf.distribute.MirrorStrategy
# as tensors are reset and the `infshape` attribute is lost
optimizer = MuSGD(learning_rate=0.1 **opt_kwargs)
```
