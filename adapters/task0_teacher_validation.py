"""Validate recorded single-episode teacher data; capacity is never valid length."""
import json
from pathlib import Path
import numpy as np
from adapters.task0_rl import ActionCoordinates, reduce_state


def validate_teacher(folder, allowed, excluded, checkpoint):
    folder = Path(folder)
    m = json.loads((folder / 'manifest.json').read_text())
    n = m['transitions']
    if not m['complete'] or m['official_success'] is not True:
        raise ValueError('Incomplete or officially unsuccessful teacher')
    if m['instance_id'] not in allowed or m['instance_id'] in excluded:
        raise ValueError('Teacher instance is not allowed train data')
    if n < 32 or m['observations'] != n+1 or m['episode_boundaries'] != [0,n]:
        raise ValueError('Invalid single episode alignment')
    arrays = {}
    for key, tail, dtype, count in [
        ('actions',(23,),np.float32,n),('raw_state',(61,),np.float32,n+1),
        ('rewards',(),np.float32,n),('dones',(),bool,n),('truncated',(),bool,n),
        *[(f'image_{i}',(224,224,3),np.uint8,n+1) for i in range(3)],
        ('timestamp',(),np.float64,n+1),('wall_timestamp',(),np.float64,n+1)]:
        x = np.load(folder/f'{key}.npy',mmap_mode='r')
        if x.shape[0] < count or x.shape[1:] != tail or x.dtype != np.dtype(dtype):
            raise ValueError(f'Invalid {key} shape/dtype/capacity')
        if not np.isfinite(x[:count]).all():
            raise ValueError(f'Nonfinite {key}')
        arrays[key] = x
    if arrays['dones'][:n-1].any() or arrays['truncated'][:n-1].any():
        raise ValueError('Interior episode boundary')
    if not arrays['dones'][n-1] or arrays['truncated'][n-1] or arrays['rewards'][n-1] != 1:
        raise ValueError('Successful terminal reward/boundary missing')
    events = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    if len(events) != n or not events[0]['episode_start'] or not events[-1]['episode_end']:
        raise ValueError('Event alignment/boundary mismatch')
    for i,e in enumerate(events):
        if e['episode_start'] != (i == 0) or e['episode_end'] != (i == n-1):
            raise ValueError('Event episode boundary mismatch')
        if e['step'] != i or e['reward'] != float(arrays['rewards'][i]) or e['terminated'] != bool(arrays['dones'][i]):
            raise ValueError('Event-action alignment mismatch')
    if not np.allclose(arrays['timestamp'][:n+1],np.arange(n+1)/30.):
        raise ValueError('Simulation timestamps mismatch')
    coordinates = ActionCoordinates.load(checkpoint)
    error = 0.
    for start in range(n-31):
        action=arrays['actions'][start:start+32]; state=reduce_state(arrays['raw_state'][start])
        error=max(error,float(np.abs(coordinates.decode(coordinates.encode(action,state),state)-action).max()))
    if error > 1e-4:
        raise ValueError(f'Encode/decode mismatch {error}')
    return {'passed':True,'instance_id':m['instance_id'],'transitions':n,'observations':n+1,
            'capacity':m['capacity'],'encode_decode_max_error':error,'folder':str(folder.resolve()),
            'successor_alignment':'indexed observation[t+1]; per-event terminal and timestamp alignment verified'}
