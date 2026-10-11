"""Ilia-native actor with the pinned RT-EXPO BASE fast-window kernel.

This is an actor-only BASE bridge, not a learned-Q EXPO result. Native stage
voting, correction rules, transforms and correlated-noise inpainting stay in
Ilia. Temporal compression is disabled to align physical prefix steps exactly.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def load_rt_base_kernel(path):
    import jax
    import jax.numpy as jnp
    source=Path(path).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_jitted_fast_select')
    namespace={'jax':jax,'jnp':jnp,'partial':partial}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_jitted_fast_select'],hashlib.sha256(source.encode()).hexdigest()


def forbidden_q(*args,**kwargs):
    raise RuntimeError('F BASE must never invoke an untrained critic/editor')


class IliaRTProxy:
    def __init__(self,policy,kernel_path,log,delay):
        if delay not in (0,4):raise ValueError('Supported declared delays:0/4')
        self.policy=policy;self.delay=delay;self.log=Path(log)
        self.kernel,self.kernel_sha=load_rt_base_kernel(kernel_path)
        self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
        self.calls=0
    def __getattr__(self,key):return getattr(self.policy,key)
    def reset(self):
        if self.future is not None:self.future.result();self.future=None
        self.calls=0;self.policy.reset()
    def arm(self,wrapper,obs):
        if not self.delay or self.future is not None or wrapper.last_actions is None:return
        remaining=len(wrapper.last_actions)-wrapper.action_index
        if remaining!=self.delay:return
        prefix=np.array(wrapper.last_actions[wrapper.action_index:],copy=True)
        inputs=wrapper.prepare_batch_for_pi_behavior(wrapper.process_obs(obs))
        inputs={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in inputs.items()}
        def slow():
            tick=time.monotonic();output=self.policy.infer(inputs,initial_actions=prefix)
            return output,time.monotonic()-tick
        self.future=self.pool.submit(slow)
    def infer(self,obs,**kwargs):
        import jax
        import jax.numpy as jnp
        tick=time.monotonic();delay=0;slow_seconds=None
        if self.future is None:output=self.policy.infer(obs,**kwargs)
        else:
            output,slow_seconds=self.future.result();self.future=None;delay=self.delay
        wait=time.monotonic()-tick;actions=np.asarray(output['actions'],dtype=np.float32)[:,:23]
        if actions.shape!=(30,23) or not np.isfinite(actions).all():raise ValueError('Invalid Ilia action chunk')
        # BASE performs no Q/edit math, so slicing the post-transform physical
        # actions is exact and does not introduce a second normalization path.
        selected=self.kernel(jax.random.key(0),jnp.asarray(actions[None]),{'camera':jnp.zeros((1,1,1,3))},jnp.zeros((1,32)),{},{},{},encoder_fn=forbidden_q,edit_fn=forbidden_q,critic_fn=forbidden_q,N=1,n_edit_samples=0,num_qs=0,num_min_qs=0,replan_steps=20,action_dim=23,state_dim=23,padded_action_dim=32,delay=delay,edit_scale=0.,edit_action_xyzg=False,freeze_encoder=True,only_base_actions=True,critic_camera_keys=('camera',))
        physical=np.asarray(selected['selected']);np.testing.assert_array_equal(physical,actions[delay:delay+20])
        # Native carry follows the selected20 physical steps, with4 real future
        # actions. No fake action padding is supplied to the native wrapper.
        output={**output,'actions':np.concatenate([physical,actions[delay+20:delay+24]])}
        self.calls+=1
        with self.log.open('a') as f:f.write(json.dumps({'call':self.calls,'delay_physical_steps':delay,'slow_seconds':slow_seconds,'boundary_wait_seconds':wait,'meets30hz_prefix_budget':slow_seconds<=delay/30 if slow_seconds is not None else None,'kernel_sha256':self.kernel_sha,'selection':'RT-EXPO upstream BASE N1; critic/editor disabled','physical_actions':20,'carry_actions':4})+'\n')
        return output
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
