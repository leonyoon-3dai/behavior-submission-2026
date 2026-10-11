"""Matched two-chunk BC draws; successful teachers supply controller-space actions."""
import json
from pathlib import Path
import numpy as np
from adapters.task0_rl import ActionCoordinates, reduce_state


class BCReplay:
    def __init__(self, replay, teachers, checkpoint, sampling="uniform"):
        self.sampling = sampling
        self.events = {}
        self.replay = replay
        self.original = [i for i,(r,_) in enumerate(replay.episodes) if r['split']=='train']
        self.rng = np.random.default_rng(0)
        self.teacher_rng = np.random.default_rng(1000)
        self.focused_rng = np.random.default_rng(2000)
        self.teacher_ids = []
        self.draws = 0
        allowed = {replay.episodes[i][0]['task_instance_id'] for i in self.original}
        excluded = {282,284,285,288,296,297,299,300,301} | {r['task_instance_id'] for r,_ in replay.episodes if r['split']=='heldout'}
        from adapters.task0_teacher_validation import validate_teacher
        coordinates = ActionCoordinates.load(checkpoint)
        instances = set()
        for folder in teachers:
            folder = Path(folder)
            validate_teacher(folder, allowed, excluded, checkpoint)
            m = json.loads((folder/'manifest.json').read_text())
            instance = m['instance_id']
            if not m['complete'] or m['official_success'] is not True or instance not in allowed or instance in excluded or instance in instances:
                raise ValueError('Teacher must be a distinct successful allowed train instance')
            instances.add(instance)
            n = m['transitions']
            if n < 32 or m['observations'] != n+1:
                raise ValueError('Invalid teacher transition alignment')
            arrays = {k:np.load(folder/f'{k}.npy', mmap_mode='r') for k in ('raw_state','actions','rewards','dones','image_0','image_1','image_2')}
            for start in (0, n//2, n-32):
                action = arrays['actions'][start:start+32]
                state = reduce_state(arrays['raw_state'][start])
                error = np.max(np.abs(coordinates.decode(coordinates.encode(action,state),state)-action))
                if not np.isfinite(error) or error > 1e-4:
                    raise ValueError(f'Teacher action encode/decode mismatch {error}')
            self.events[len(replay.episodes)] = [json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
            self.teacher_ids.append(len(replay.episodes))
            replay.episodes.append(({'split':'train','transitions':n,'episode_index':f'teacher-instance-{instance}',
                'task_instance_id':instance,'folder':str(folder),'teacher':True},arrays))

    def location(self, rng, pool, focused_rng=None):
        episode = int(rng.choice(pool))
        length = self.replay.episodes[episode][0]['transitions']
        # Consume exactly the uniform episode/start draws in both conditions.
        # A separate RNG changes only the teacher segment, keeping episodes matched.
        start = int(rng.integers(0,length-32+1))
        if episode in self.teacher_ids and self.sampling == 'contact-focused':
            focused_rng = self.focused_rng if focused_rng is None else focused_rng
            if focused_rng.random() < .5:
                contacts = [i for i,e in enumerate(self.events[episode]) if e.get('contact_hold_seconds',0) > 0]
                if contacts:
                    step = int(focused_rng.choice(contacts))
                    start = int(focused_rng.integers(max(0,step-31), min(step,length-32)+1))
        return episode, start

    def batch(self, actor, sampling_log):
        locations = [self.location(self.rng,self.original) for _ in range(2)]
        # Consume the same original draws for both conditions; replace exactly one for B2.
        if self.teacher_ids:
            locations[1] = self.location(self.teacher_rng,self.teacher_ids)
        with Path(sampling_log).open('a') as stream:
            stream.write(json.dumps([self.describe(e,s) for e,s in locations])+'\n')
        self.draws += 2
        return self.replay.batch(actor, locations, discount=.999)

    def describe(self, episode, start):
        record=self.replay.episodes[episode][0]; n=record['transitions']
        events=self.events.get(episode,[])[start:start+32]
        return {'episode':record['episode_index'],'instance_id':record['task_instance_id'],
                'source':'teacher' if episode in self.teacher_ids else 'original',
                'physical_events_measured':episode in self.teacher_ids,
                'start':start,'end':start+31,'valid_action_ratio':1.,'padding':False,
                'terminal_action_included':start+32==n,
                'last_128_step_proxy_overlap':start+31>=n-128,
                'first_third_time_proxy_overlap':start<n//3,
                'native_grasp_overlap':any(any(v=='1' for v in e.get('grasp_radio',{}).values()) for e in events),
                'measured_contact_overlap':any(e.get('contact_hold_seconds',0)>0 for e in events),
                'successful_event_overlap':any(e.get('official_success') is True for e in events)}

    def dry_run(self, output, steps):
        # Independent RNGs reproduce actual initial draws without consuming training state.
        original=np.random.default_rng(0); teacher=np.random.default_rng(1000);focused=np.random.default_rng(2000)
        rows=[]
        for _ in range(steps):
            locations=[self.location(original,self.original) for _ in range(2)]
            if self.teacher_ids: locations[1]=self.location(teacher,self.teacher_ids,focused)
            rows.extend(self.describe(e,s) for e,s in locations)
        Path(output).write_text(json.dumps({'sampling':self.sampling,'draws':rows,
            'physical_event_fields':['native_grasp_overlap','measured_contact_overlap','successful_event_overlap'],
            'time_proxy_fields':['last_128_step_proxy_overlap','first_third_time_proxy_overlap'],
            'all_chunks_full_length':True,
            'expected_terminal_draws_per_teacher':{str(self.replay.episodes[e][0]['episode_index']):steps/len(self.teacher_ids)/(self.replay.episodes[e][0]['transitions']-31) for e in self.teacher_ids},
            'expected_teacher_terminal_draws':
             sum(steps/len(self.teacher_ids)/(self.replay.episodes[e][0]['transitions']-31) for e in self.teacher_ids) if self.teacher_ids else 0},indent=2)+'\n')
