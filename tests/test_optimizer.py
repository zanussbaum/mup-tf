import numpy as np
import pytest
import tensorflow as tf

from mup_tf import InfDim, InfShape, MuAdam, MuSGD


@pytest.mark.parametrize(
    "n_steps, amsgrad",
    [(1, False), (5, False), (10, False), (1, True), (5, True), (10, True)],
)
def test_adam(n_steps, amsgrad):
    infshape = InfShape([InfDim(1, 1), InfDim(1, 1)])

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuAdam(learning_rate=0.1, amsgrad=amsgrad, infshapes={var.name: infshape})
    mu_vals = []
    for _ in range(n_steps):
        opt.minimize(loss_fn, [var])
        mu_vals.append(var.numpy())

    var = tf.Variable(10.0)

    opt = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=amsgrad)

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    vals = []
    for _ in range(n_steps):
        opt.minimize(loss_fn, [var])
        vals.append(var.numpy())

    assert tf.math.reduce_all(tf.equal(mu_vals, vals))


def test_adam_with_inf():
    # width mult should be based on last dim, so 4
    infshape = InfShape([InfDim(4, 8), InfDim(4, 16)])

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuAdam(learning_rate=0.1, infshapes={var.name: infshape})
    opt.minimize(loss_fn, [var])

    val = var.numpy()

    # lr scaled by width mult (4)
    # first step is -learning_rate * sign(grad)
    assert np.isclose(val, 10.0 - (0.1 / 4.0))


@pytest.mark.parametrize(
    "n_steps, momentum, nesterov",
    [
        (1, 0.9, False),
        (5, 0.9, False),
        (10, 0.9, False),
        (1, 0.9, True),
        (5, 0.9, True),
        (10, 0.9, True),
    ],
)
def test_sgd(n_steps, momentum, nesterov):
    infshape = InfShape([InfDim(1, 1), InfDim(1, 1)])

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuSGD(
        learning_rate=0.1,
        momentum=momentum,
        nesterov=nesterov,
        infshapes={var.name: infshape},
    )
    mu_vals = []
    for _ in range(n_steps):
        opt.minimize(loss_fn, [var])
        mu_vals.append(var.numpy())

    var = tf.Variable(10.0)

    opt = tf.keras.optimizers.SGD(
        learning_rate=0.1, momentum=momentum, nesterov=nesterov
    )

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    vals = []
    for _ in range(n_steps):
        opt.minimize(loss_fn, [var])
        vals.append(var.numpy())

    assert tf.math.reduce_all(tf.equal(mu_vals, vals))


def test_sgd_with_inf():
    # since ninf == 2, scaled by ratio of 4/2 = 2
    infshape = InfShape([InfDim(4, 8), InfDim(4, 16)])

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuSGD(learning_rate=0.1, infshapes={var.name: infshape})
    opt.minimize(loss_fn, [var])

    val = var.numpy()

    # lr scaled by width mult 2
    # first step is -learning_rate * grad
    assert tf.math.reduce_all(tf.equal(val, 10.0 - 10 * (0.1 / 2.0)))

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuSGD(
        learning_rate=0.1, momentum=0.9, nesterov=True, infshapes={var.name: infshape}
    )
    opt.minimize(loss_fn, [var])

    val = var.numpy()

    # lr scaled by width mult 2
    # v = momentum * v - learning_rate * grad
    # where v is init at 0
    # first step is -learning_rate * grad * (1 + momentum)
    update_lr = 0.1 / 2.0
    grad = 10.0
    mu = 0.9
    assert np.isclose(val, 10.0 - (update_lr * grad * (1 + mu)))


def test_ninf_one_inf():
    infshape = InfShape([InfDim(None, 4), InfDim(1, 10)])

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuAdam(learning_rate=0.1, infshapes={var.name: infshape})

    opt.minimize(loss_fn, [var])

    val = var.numpy()

    # no change for adam
    assert np.isclose(val, 10.0 - 0.1), f"Expected: {10.0 - 0.1}\t Actual: {val}"

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuSGD(learning_rate=0.1, infshapes={var.name: infshape})

    opt.minimize(loss_fn, [var])

    val = var.numpy()

    assert np.isclose(val, 10.0 - 10 * (0.1 * 10)), f"Expected: {10.0 - 10 * (0.1 * 10)}\t Actual: {val}"

    var = tf.Variable(10.0)
    var.infshape = infshape

    # d(loss)/d(var) = var
    def loss_fn():
        return (var**2) / 2.0

    opt = MuSGD(
        learning_rate=0.1, momentum=0.9, nesterov=True, infshapes={var.name: infshape}
    )

    opt.minimize(loss_fn, [var])

    val = var.numpy()
    update_lr = 0.1 * 10
    grad = 10.0
    mu = 0.9
    assert np.isclose(val, 10.0 - (update_lr * grad * (1 + mu))), f"Expected: {10.0 - (update_lr * grad * (1 + mu))}\t Actual: {val}"
