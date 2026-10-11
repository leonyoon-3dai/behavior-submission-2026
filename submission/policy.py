"""Load the preserved native actor or task-specific RT route without training."""
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for path in ('adapters','external/1st_behavior-1k-solution/src','external/1st_behavior-1k-solution/openpi/src',
             'external/1st_behavior-1k-solution/openpi/packages/openpi-client/src','external/expo-ft-rt'):
    sys.path.insert(0,str(ROOT/path))

def factory(task,route,*,model_root,log_root,seed=0):
    import jax
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper,B1KWrapperConfig
    checkpoint=Path(model_root)/route['actor']
    condition=route['condition']
    if condition=='A0-original':
        from b1k.policies.policy_config import create_trained_policy
        from b1k.training.config import get_config
        policy=create_trained_policy(get_config('pi_behavior_b1k_fast'),checkpoint,sample_kwargs={'num_steps':20})
        policy._rng=jax.random.key(seed)
        config=B1KWrapperConfig(actions_to_execute=26,actions_to_keep=4,execute_in_n_steps=20)
    elif condition=='RT-EXPO':
        from adapters.ilia_expo_learner import build,load_auxiliary
        from adapters.ilia_expo_inference import IliaExpoProxy
        agent,actor,native,_=build(checkpoint,critic_checkpoint=Path(model_root)/route['critic_assets'],critic_asset=route['task_name'])
        tree=Path(model_root)/route['auxiliary']
        if (tree/'inference-params.npz').exists():
            from submission.auxiliary import load_inference
            agent=load_inference(agent,tree/'inference-params.npz')
        else: agent=load_auxiliary(agent,tree)
        log=Path(log_root)/f'task{task:03d}-rt-inference.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        policy=IliaExpoProxy(agent,actor,native,'RT-EXPO',log,seed)
        config=B1KWrapperConfig(actions_to_execute=15,actions_to_keep=4,execute_in_n_steps=15)
    else:raise ValueError(f'Unreviewed condition {condition}')
    class Wrapper(B1KPolicyWrapper):
        def act(self,obs):
            if condition=='RT-EXPO':self.policy.arm(self,obs)
            return super().act(obs)
        def reset(self):
            super().reset()
            if condition=='A0-original':self.policy._rng=jax.random.key(seed)
        def close(self):
            if condition=='RT-EXPO':
                self.policy.close()
                # Jitted static actor references can outlive Python garbage collection.
                # This route owns its model buffers; release them after its worker joins.
                seen=set()
                for leaf in jax.tree_util.tree_leaves((self.policy.agent,self.policy.actor.model_state)):
                    if isinstance(leaf,jax.Array) and id(leaf) not in seen:
                        seen.add(id(leaf))
                        if not leaf.is_deleted():leaf.delete()
            self.base_policy=None;self.policy=None
            gc.collect();jax.clear_caches();gc.collect()
    return Wrapper(policy,task_id=task,config=config)
