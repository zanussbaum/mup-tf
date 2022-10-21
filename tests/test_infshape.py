import pytest

from mup_tf import InfDim, InfShape, zip_infshape


def test_is_finite():
    assert InfDim(None, 10).isinf() is False
    assert InfDim(None, None).isinf() is False
    assert InfDim(10, None).isinf() is True


def test_width_mult():
    dim = InfDim(10, None)

    with pytest.raises(ValueError):
        dim.width_mult()

    assert InfDim(1, 10).width_mult() == 10

    assert InfDim(None, 10).width_mult() == 1


def test_width_mult_infshape():
    infshape = InfShape([InfDim(None, 100), InfDim(128, 1024), InfDim(64, 128)])

    assert infshape.ninf() == 2

    assert infshape.width_mult() == (128 / 64)
    assert infshape.base_shape() == [None, 128, 64]


def test_zip_infshape():
    shape = zip_infshape([64, 128, 1024], [32, 128, 2048])

    assert shape.ninf() == 2

    assert shape.width_mult() == 2048 / 1024
    assert shape.base_shape() == [64, None, 1024]

    shape = zip_infshape([InfDim(64, 512), 128, 1024], [32, 128, 2048])

    assert shape.ninf() == 2

    assert shape.width_mult() == 2048 / 1024
    assert shape.base_shape() == [64, None, 1024]
