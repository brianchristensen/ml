# temper_jax/trainer.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from torchvision import datasets, transforms
from flax.training import train_state
from flax import struct
from flax.core import FrozenDict
from typing import Any, Callable, Tuple
from model import TemperGraph


class TrainState(train_state.TrainState):
    model: nn.Module = struct.field(pytree_node=False)
    batch_stats: FrozenDict[str, Any] = struct.field(default_factory=FrozenDict)

def prepare_dataset(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        transforms.Lambda(lambda x: x.numpy()),
    ])

    dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    def gen():
        for i in range(0, len(dataset), batch_size):
            batch = dataset.data[i:i + batch_size].float().div(255).reshape(-1, 784).numpy()
            labels = dataset.targets[i:i + batch_size].numpy()
            yield (batch, labels)

    return gen()


def compute_loss(params, model, batch, rng):
    x, _ = batch
    variables = {"params": params}
    latent, pred_next = model.apply(variables, x, rng)
    pred_error = jnp.mean((pred_next - x) ** 2)
    usage_reward = 0.0  # (stub: could add usage tracking later)
    return pred_error - 0.01 * usage_reward, pred_error


@jax.jit
def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], rng) -> Tuple[TrainState, float, float]:
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, pred_error), grads = grad_fn(state.params, state.model, batch, rng)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, pred_error


def train(num_epochs: int = 5, batch_size: int = 64):
    rng = jax.random.PRNGKey(0)

    # Build model
    model = TemperGraph(input_dim=784, hidden_dim=8, num_tempers=4, max_hops=8)

    # Dummy input for init
    x_dummy = jnp.ones((batch_size, 784))
    variables = model.init(rng, x_dummy, rng)

    # Initialize train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3),
        model=model,
    )

    train_ds = prepare_dataset(batch_size)
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_error = 0.0
        steps = 0

        for batch in train_ds:
            rng, subrng = jax.random.split(rng)
            state, loss, error = train_step(state, batch, subrng)
            total_loss += loss
            total_error += error
            steps += 1

        avg_loss = total_loss / steps
        avg_error = total_error / steps
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | MSE: {avg_error:.4f}")


if __name__ == "__main__":
    train()
