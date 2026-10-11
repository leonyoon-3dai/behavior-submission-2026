"""Restore inference parameters only; optimizer payloads remain in research backups."""
import json
from pathlib import Path
import numpy as np
PARTS=('critic','target_critic','filter_critic','batch_encoder','edit_actor','temp')

def save_inference(tree,path):
    import jax
    arrays={};meta={}
    for name in PARTS:
        leaves,_=jax.tree.flatten(tree[name]['params'])
        meta[name]=[]
        for i,value in enumerate(leaves):
            key=f'{name}_{i}';arr=np.asarray(value);dtype=arr.dtype.name
            arrays[key]=arr.view(np.uint16) if dtype=='bfloat16' else arr
            meta[name].append({'key':key,'dtype':dtype,'shape':list(arr.shape)})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**arrays);path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')

def load_inference(agent,path):
    import jax
    import ml_dtypes
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text());changes={}
    with np.load(path,allow_pickle=False) as archive:
        for name in PARTS:
            old=getattr(agent,name);leaves,structure=jax.tree.flatten(old.params)
            if len(meta[name])!=len(leaves):raise ValueError(f'Incompatible auxiliary tree {name}')
            restored=[]
            for item,template in zip(meta[name],leaves):
                arr=archive[item['key']]
                if item['dtype']=='bfloat16':arr=arr.view(ml_dtypes.bfloat16)
                if arr.shape!=template.shape or arr.dtype.name!=np.asarray(template).dtype.name:
                    raise ValueError(f'Incompatible auxiliary leaf {item["key"]}')
                restored.append(arr.copy())
            changes[name]=old.replace(params=jax.tree.unflatten(structure,restored))
    return agent.replace(**changes).cache_infer_params()
