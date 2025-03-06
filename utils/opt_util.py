from optax._src.alias import *  # lazy import
import jax
import jax.numpy as jnp
from flax import struct


def sgd(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """
    Stochastic Gradient Descent (SGD) with L2 regularization optimizer.

    This is the standard SGD optimizer with optional momentum and weight decay.
    Weight decay is applied before the momentum step. 
    
    Args:
        learning_rate (base.ScalarOrSchedule): The learning rate or learning rate schedule.
        momentum (Optional[float]): The momentum coefficient. If None, no momentum is applied.
        nesterov (bool): Whether to use Nesterov momentum. Default is False.
        weight_decay (float): The weight decay coefficient. Default is 1e-4.
        mask (Optional[Union[Any, Callable[[base.Params], Any]]]): A mask to apply to the weights.
            It can be a constant or a callable that takes the model parameters and returns a mask.
            Default is None.
        accumulator_dtype (Optional[Any]): The data type for the accumulator. Default is None.

    Returns:
        base.GradientTransformation: The gradient transformation function.

    """
    return combine.chain(
        transform.add_decayed_weights(weight_decay, mask),
        (transform.trace(decay=momentum, nesterov=nesterov,
                         accumulator_dtype=accumulator_dtype)
         if momentum is not None else base.identity()),
        transform.scale_by_learning_rate(learning_rate)
    )


def sgdw(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """
    Stochastic Gradient Descent with *Decoupled** Weight Decay (SGDW) optimizer.

    Weight decay is applied after the momentum step.

    References:
        Loshchilov et al, `Decoupled Weight Decay Regularization
        <https://arxiv.org/abs/1711.05101>`_, 2019

    Args:
        learning_rate (base.ScalarOrSchedule): The learning rate or learning rate schedule.
        momentum (Optional[float]): The momentum coefficient. If None, no momentum is applied.
        nesterov (bool): Whether to use Nesterov momentum. Default is False.
        weight_decay (float): The weight decay coefficient. Default is 1e-4.
        mask (Optional[Union[Any, Callable[[base.Params], Any]]]): A mask to apply to the weights.
            It can be a constant or a callable that takes the model parameters and returns a mask.
            Default is None.
        accumulator_dtype (Optional[Any]): The data type for the accumulator. Default is None.

    Returns:
        base.GradientTransformation: The gradient transformation function.

    """
    return combine.chain(
        (transform.trace(decay=momentum, nesterov=nesterov,
                         accumulator_dtype=accumulator_dtype)
         if momentum is not None else base.identity()),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate)
    )

# Code is taken from: https://github.com/google-research/sam/blob/main/sam_jax/efficientnet/optim.py#L94
@struct.dataclass
class ExponentialMovingAverage:
  """Exponential Moving Average as implemented in Tensorflow."""

  # Moving average of the parameters.
  params_ema: Any
  # Decay to use for the update (typical values are 0.999, 0.9999, etc...).
  decay: float
  # For how many steps we should just keep the new parameters instead of an
  # average (useful if we don't want the initial weights to be included in the
  # average).
  warmup_steps: int

  def update_moving_average(self, new_target: Any,
                            step: jnp.ndarray) -> Any:
    """Updates the moving average of the target.
    Args:
      new_target: New values of the target (example: weights of a network
        after gradient step).
      step: Current step (used only for warmup).
    Returns:
      The updated ExponentialMovingAverage.
    """

    factor = jnp.float32(step >= self.warmup_steps)
    delta = step - self.warmup_steps
    decay = jnp.minimum(self.decay, (1. + delta) / (10. + delta))
    decay *= factor

    # decay = self.decay
    
    weight_ema = jax.tree.map(
        lambda a, b: (1 - decay) * a + decay * b, new_target, self.params_ema)
    return self.replace(params_ema=weight_ema)