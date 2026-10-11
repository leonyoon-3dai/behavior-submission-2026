"""Official batched RGB/proprio boundary; privileged evaluator fields are discarded."""
import numpy as np

STATE = 'robot_r1::proprio'
CAMERAS = (
 'robot_r1::robot_r1:zed_link:Camera:0::rgb',
 'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb',
 'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb',
)
TASKS = {0:'turning_on_radio',1:'picking_up_trash',2:'putting_away_Halloween_decorations',
         3:'cleaning_up_plates_and_food',5:'setting_mousetraps',7:'picking_up_toys'}

def observation(obs):
    state=np.asarray(obs[STATE])
    if state.shape != (1,61) or not np.isfinite(state).all():
        raise ValueError(f'Expected finite official proprio (1,61), got {state.shape}')
    task=np.asarray(obs['task_id'])
    if task.shape not in ((1,), (1,1)) or task.dtype.kind not in 'iu':
        raise ValueError('Expected integer task_id for one environment: (1,) or official (1,1)')
    task=task.reshape(1)
    result={STATE:state[0].copy(), 'task_id':task.copy()}
    for key in CAMERAS:
        img=np.asarray(obs[key])
        if img.ndim!=4 or img.shape[0]!=1 or img.shape[-1] not in (3,4) or img.dtype!=np.uint8:
            raise ValueError(f'Expected batched uint8 RGB/RGBA for {key}: {img.shape}/{img.dtype}')
        result[key]=np.array(img[0,...,:3],copy=True)
    return int(task[0]), result

def action_batch(value):
    value=np.asarray(value,dtype=np.float32)
    if value.shape != (23,) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite native23D action: {value.shape}')
    return value[None].copy()

class TaskRouter:
    """Single environment and one active policy. Factory loads actor+aux+assets atomically."""
    def __init__(self,routes,factory):
        self.routes={int(k):v for k,v in routes.items()}
        self.factory=factory;self.active=None;self.task=None
    def act(self,obs):
        task,native=observation(obs)
        if task not in self.routes:
            # Unsupported tasks are explicitly documented as no-op, never inferred with a wrong embedding.
            self.reset();return np.zeros((1,23),np.float32)
        if task!=self.task:
            self.close()
            self.active=self.factory(task,self.routes[task]);self.task=task
        result=self.active.act(native)
        if hasattr(result,'detach'): result=result.detach().cpu().numpy()
        return action_batch(result)
    def reset(self):
        if self.active is not None:self.active.reset()
    def close(self):
        if self.active is not None:
            self.active.reset()
            if hasattr(self.active,'close'):self.active.close()
            self.active=None
        self.task=None
