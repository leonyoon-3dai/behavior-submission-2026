"""Read-only heldout diagnostics; terminal proximity is not a semantic press label."""
from pathlib import Path
import jax
import numpy as np
from flax import nnx
from adapters.task0_replay import Task0Replay
from adapters.task0_rl import IMAGE_NAMES
from expo_ft.agents.alg.batch_utils import prepare_critic_batch


def validate_checkpoint(agent, actor, replay_root: Path, discount, device_sharding):
    replay = Task0Replay(replay_root)
    groups = {'common': [], 'near_terminal': []}
    for index, (record, _) in enumerate(replay.episodes):
        if record['split'] != 'heldout':
            continue
        length = record['transitions']
        groups['common'].extend((index, int((length-32)*fraction)) for fraction in (.25,.75))
        groups['near_terminal'].extend((index, length-offset) for offset in (64,32))
    if any(not locations for locations in groups.values()):
        raise ValueError('Heldout episodes required')

    @jax.jit
    def loss_fn(state, data):
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        return model.compute_loss(jax.random.PRNGKey(17),data[0],data[1],train=False).mean()

    result = {}
    for name, locations in groups.items():
        losses = []
        for start in range(0,len(locations),2):
            batch = jax.device_put(replay.batch(actor,locations[start:start+2],discount=discount),device_sharding)
            prepared = prepare_critic_batch(batch,32,23,23,32,8,IMAGE_NAMES)
            losses.append(float(jax.device_get(loss_fn(agent.actor_train_state,actor.prepare_batch_for_actor(prepared)))))
        if not np.isfinite(losses).all():
            raise ValueError('Nonfinite heldout loss')
        result[name] = {'loss':float(np.mean(losses)), 'batch_losses':losses,
            'locations':[[replay.episodes[index][0]['episode_index'],start] for index,start in locations],
            'chunks':len(locations)}
    return {'groups':result,'rng_seed':17,'batch_size':2,'training_updates_performed':0,
        'note':'Near-terminal uses final 64/32-step starts. These are temporal locations, not annotated press segments.'}
