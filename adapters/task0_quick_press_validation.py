"""Time-bounded, read-only semantic-stage loss and sampled-action diagnostics."""
import ast
import hashlib
import inspect
import json
import textwrap
import time
from pathlib import Path
import jax
import numpy as np
from flax import nnx
from adapters.task0_replay import Task0Replay
from adapters.task0_rl import IMAGE_NAMES,ActionCoordinates
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from openpi.models.pi0 import Pi0

ROOT=Path(__file__).resolve().parents[1]
FM_SEEDS=(17,29,43,61,83)
PARTS={'left_arm':slice(7,14),'right_arm':slice(15,22),'left_gripper':slice(14,15),'right_gripper':slice(22,23)}


def dimension_loss_function():
    source=inspect.getsource(Pi0.compute_loss)
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    fn.name='dimension_loss';fn.decorator_list=[]
    final=fn.body[-1]
    assert isinstance(final,ast.Return) and isinstance(final.value,ast.Call)
    assert ast.unparse(final.value.func)=='jnp.mean' and ast.unparse(final.value.args[0].func)=='jnp.square'
    final.value=final.value.args[0]
    namespace=dict(Pi0.compute_loss.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<diagnostic_dimension_loss>','exec'),namespace)
    return namespace['dimension_loss'],hashlib.sha256(source.encode()).hexdigest()


def validate_quick_press(agent,actor,replay_root,discount,device_sharding,report_path,deadline):
    started=time.time();replay=Task0Replay(replay_root)
    audit_path=ROOT/'experiments/task0/dataset-analysis/annotation-summary.json'
    annotations={r['episode_index']:r for r in json.loads(audit_path.read_text())['episodes']}
    locations=[]
    def add(index,start,stage,progress=None):
        record=replay.episodes[index][0]
        assert record['split']=='heldout' and 0<=start<=record['transitions']-32
        locations.append({'replay_index':index,'episode_index':record['episode_index'],'start':start,'stage':stage,'progress':progress})
    for stage in ('common','move_to','pick_up','press'):
        for index,(record,_) in enumerate(replay.episodes):
            if record['split']!='heldout':continue
            annotation=annotations[record['episode_index']];length=record['transitions']
            assert annotation['first_terminal']+1==length
            if stage=='common':
                for fraction in (.25,.75):add(index,int((length-32)*fraction),stage)
            elif stage in ('move_to','pick_up'):
                segment=annotation['segments'][0 if stage=='move_to' else 1]
                lo,hi=segment[1],min(segment[2],length)
                if hi-lo<32:continue
                for fraction in (.25,.75):add(index,lo+int((hi-lo-32)*fraction),stage)
            else:
                lo=annotation['press_start'];end=min(annotation['press_end'],length)
                starts=list(range(lo,end-31,8))
                if starts[-1]!=end-32:starts.append(end-32)
                for start in starts:add(index,start,stage,(start-lo)/max(1,end-32-lo))
    assert sum(r['stage']=='press' for r in locations)==137
    raw_fn,source_sha=dimension_loss_function()
    @jax.jit
    def losses(state,data,key):
        model=nnx.merge(state.model_def,state.params);model.eval()
        return raw_fn(model,key,data[0],data[1],train=False)
    @jax.jit
    def reference(state,data):
        model=nnx.merge(state.model_def,state.params);model.eval()
        return model.compute_loss(jax.random.PRNGKey(17),data[0],data[1],train=False).mean()
    @jax.jit
    def predictions(state,observation):
        model=nnx.merge(state.model_def,state.params);model.eval()
        return model.sample_actions(jax.random.PRNGKey(101),observation,num_steps=10,num_samples=1,train=False)
    coords=ActionCoordinates.load(ROOT/'checkpoints/behave/turning_on_radio/behave_pi05_turn_on_the_radio')
    rows=[];common_losses=[];reference_delta=None;max_decode_error=0.0
    raw_path=report_path.with_suffix('.chunks.jsonl');raw_path.parent.mkdir(parents=True,exist_ok=True)
    with raw_path.open('w') as stream:
        for begin in range(0,len(locations),2):
            if time.time()>deadline:break
            selected=locations[begin:begin+2];padded=selected if len(selected)==2 else selected*2
            batch=jax.device_put(replay.batch(actor,[(r['replay_index'],r['start']) for r in padded],discount=discount),device_sharding)
            data=actor.prepare_batch_for_actor(prepare_critic_batch(batch,32,23,23,32,8,IMAGE_NAMES))
            errors=[]
            for seed in FM_SEEDS:errors.append(np.asarray(jax.device_get(losses(agent.actor_train_state,data,jax.random.PRNGKey(seed)))))
            if reference_delta is None:
                reference_delta=abs(float(errors[0].mean())-float(jax.device_get(reference(agent.actor_train_state,data))))
                assert reference_delta<1e-6,reference_delta
            if selected[0]['stage']=='common':common_losses.append(float(errors[0].mean()))
            fm=np.mean(errors,axis=0)
            pred=np.asarray(jax.device_get(predictions(agent.actor_train_state,data[0])))
            target=np.asarray(jax.device_get(data[1]))
            assert pred.shape==target.shape and pred.shape[1:]==(32,32),(pred.shape,target.shape)
            for i,r in enumerate(selected):
                arrays=replay.episodes[r['replay_index']][1];anchor=np.asarray(arrays['state'][r['start']])
                absolute=np.asarray(arrays['actions'][r['start']:r['start']+32])
                decoded_target=coords.decode(target[i],anchor)
                decode_error=float(np.max(np.abs(decoded_target-absolute)));max_decode_error=max(max_decode_error,decode_error)
                assert decode_error<1e-4,decode_error
                decoded_pred=coords.decode(pred[i],anchor);error=np.abs(pred[i,:,:23]-target[i,:,:23]);native=np.abs(decoded_pred-absolute)
                metrics={'fm_all32':float(fm[i].mean()),'fm_valid23':float(fm[i,:,:23].mean()),'fm_padding9':float(fm[i,:,23:].mean()),
                    'action_mae_first8_valid23':float(error[:8].mean()),'action_mae_full32_valid23':float(error.mean())}
                for name,dims in PARTS.items():
                    metrics['fm_'+name]=float(fm[i,:,dims].mean())
                    metrics['action_mae_first8_'+name]=float(error[:8,dims].mean())
                    metrics['action_mae_full32_'+name]=float(error[:,dims].mean())
                    metrics['native_mae_first8_'+name]=float(native[:8,dims].mean())
                assert all(np.isfinite(v) for v in metrics.values())
                row={**r,'metrics':metrics};rows.append(row)
                stream.write(json.dumps({**row,'native_mae_first8_per_channel':native[:8].mean(0).tolist()})+'\n');stream.flush()
            if begin%20==0:print(json.dumps({'diagnostic_chunks':len(rows),'planned_chunks':len(locations),'elapsed_seconds':time.time()-started}),flush=True)
    def group_name(r):
        if r['stage']!='press':return [r['stage']]
        phase='early' if r['progress']<1/3 else 'middle' if r['progress']<2/3 else 'late'
        return ['press_full','press_'+phase]
    names=sorted({name for r in rows for name in group_name(r)})
    groups={}
    for name in names:
        picked=[r for r in rows if name in group_name(r)];episodes={}
        for episode in sorted({r['episode_index'] for r in picked}):
            subset=[r for r in picked if r['episode_index']==episode]
            episodes[str(episode)]={'chunks':len(subset),'metrics':{key:float(np.mean([r['metrics'][key] for r in subset])) for key in subset[0]['metrics']}}
        groups[name]={'chunks':len(picked),'episodes':episodes,'macro_metrics':{key:float(np.mean([e['metrics'][key] for e in episodes.values()])) for key in picked[0]['metrics']}}
    return {'status':'complete' if len(rows)==len(locations) else 'time_limited_partial','groups':groups,'observed_chunks':len(rows),'planned_chunks':len(locations),
        'planned_press_chunks':137,'common_seed17_loss':float(np.mean(common_losses)) if common_losses else None,
        'fm_seeds':FM_SEEDS,'inference_seed':101,'denoise_steps':10,'batch_size':2,'dimension_mean_reference_error':reference_delta,
        'target_decode_max_error':max_decode_error,'elapsed_seconds':time.time()-started,'training_updates_performed':0,
        'raw_chunk_file':str(raw_path.resolve().relative_to(ROOT)),'raw_chunk_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'annotation_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'upstream_compute_loss_sha256':source_sha,
        'note':'Stage thirds are temporal annotation partitions, not measured contact events. Macro metrics weight each episode equally; overlapping chunks are not independent. Action errors use one fixed offline inference seed, not closed-loop success.'}
