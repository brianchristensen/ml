import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

class OperatorBank(nn.Module):
    hidden_dim: int
    embed_dim: int
    num_ops: int

    def setup(self):
        self.dense1 = nn.DenseGeneral(features=(self.num_ops, self.hidden_dim), axis=-1)
        self.dense2 = nn.DenseGeneral(features=(self.num_ops, self.hidden_dim), axis=-1)

    def __call__(self, x_aug, op_idx):
        B = x_aug.shape[0]
        h1 = nn.relu(self.dense1(x_aug))  # (B, num_ops, H)
        h2 = nn.relu(self.dense2(h1))      # (B, num_ops, H)

        idx_expand = op_idx[:, None, None]
        idx_expand = jnp.broadcast_to(idx_expand, (B, 1, self.hidden_dim))
        out = jnp.take_along_axis(h2, idx_expand, axis=1).squeeze(1)
        return out

class Temper(nn.Module):
    hidden_dim: int
    num_ops: int = 3

    def setup(self):
        self.embed_dim = self.hidden_dim // 2
        self.op_emb = self.param("op_emb", nn.initializers.normal(), (self.num_ops, self.embed_dim))
        self.op_logits = self.param("op_logits", nn.initializers.zeros, (self.num_ops,))
        self.hidden_state = self.variable("state", "hidden", lambda: jnp.zeros(self.hidden_dim))
        self.id_embeds = self.param("id_embeds", nn.initializers.normal(), (128, 4))  # up to 128 IDs

    @nn.compact
    def __call__(self, x, rng, temper_ids):
        B = x.shape[0]
        probs = nn.softmax(self.op_logits)
        op_idx = jax.random.categorical(rng, jnp.log(probs), shape=(B,))

        id_emb = self.id_embeds[temper_ids]
        op_emb = self.op_emb[op_idx]

        x_aug = jnp.concatenate([x, id_emb, op_emb], axis=-1)

        h = nn.relu(nn.Dense(self.hidden_dim)(x_aug))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))

        self.hidden_state.value = 0.95 * self.hidden_state.value + 0.05 * h.mean(axis=0)
        return h

class RoutingPolicy(nn.Module):
    input_dim: int
    num_tempers: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(self.num_tempers + 1)(x)

class PredictiveHead(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)

class TemperGraph(nn.Module):
    input_dim: int
    hidden_dim: int
    num_tempers: int
    max_hops: int = 4

    def setup(self):
        self.temper = Temper(hidden_dim=self.hidden_dim, num_ops=3)
        self.routing = RoutingPolicy(self.hidden_dim + 4, self.num_tempers)
        self.predict = PredictiveHead(self.hidden_dim, self.input_dim)
        self.tid_emb = self.param("tid_emb", nn.initializers.normal(), (self.num_tempers, 4))

        # 🔥 NEW: Force param creation for Temper
        dummy_x = jnp.ones((1, self.hidden_dim))
        dummy_ids = jnp.zeros((1,), dtype=jnp.int32)
        dummy_rng = self.make_rng("params")  # Important: this is the correct way to pull params rng
        _ = self.temper.init(dummy_rng, dummy_x, dummy_rng, dummy_ids)

    def __call__(self, x, rng):
        B, D = x.shape

        pad = (self.hidden_dim - D % self.hidden_dim) % self.hidden_dim
        if pad > 0:
            x = jnp.pad(x, ((0, 0), (0, pad)))

        num_patches = x.shape[1] // self.hidden_dim
        patches = x.reshape(B * num_patches, self.hidden_dim)

        patch_states = patches
        patch_tempers = jax.random.randint(rng, (patches.shape[0],), 0, self.num_tempers)
        patch_done = jnp.zeros((patches.shape[0],), dtype=bool)

        def cond_fn(carry):
            _, _, patch_done, _, hop = carry
            return (hop < self.max_hops) & (jnp.any(~patch_done))

        def body_fn(carry):
            patch_states, patch_tempers, patch_done, rng, hop = carry
            rng, subrng = jax.random.split(rng)

            active_mask = ~patch_done
            active_idx = jnp.where(active_mask, size=patch_states.shape[0], fill_value=0)[0]

            x_active = patch_states[active_idx]
            t_active = patch_tempers[active_idx]
            rng_keys = jax.random.split(subrng, active_idx.shape[0])

            # ✅ Direct call to the Temper submodule
            updated_x = self.temper(x_active, rng=rng_keys, temper_ids=t_active)

            # Merge back
            patch_states = patch_states.at[active_idx].set(updated_x)

            enriched = jnp.concatenate([updated_x, self.tid_emb[t_active]], axis=-1)
            logits = self.routing(enriched)
            probs = nn.softmax(logits)
            rng, sample_rng = jax.random.split(rng)
            sampled = jax.random.categorical(sample_rng, jnp.log(probs))
            stop = sampled == self.num_tempers
            next_tempers = jnp.clip(sampled, 0, self.num_tempers - 1)

            patch_tempers = patch_tempers.at[active_idx].set(next_tempers)
            patch_done = patch_done.at[active_idx].set(stop)

            return (patch_states, patch_tempers, patch_done, rng, hop + 1)

        carry = (patch_states, patch_tempers, patch_done, rng, 0)
        patch_states, patch_tempers, patch_done, rng, _ = jax.lax.while_loop(cond_fn, body_fn, carry)

        patch_states = patch_states.reshape(B, num_patches, self.hidden_dim)
        latent = patch_states.mean(axis=1)
        pred = self.predict(latent)
        return latent, pred
