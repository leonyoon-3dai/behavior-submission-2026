"""Simulator-side transition recording: actual reward and real successor RGB/state."""
import json
from pathlib import Path
import cv2
import numpy as np


def array(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)


class OnlineReplayWriter:
    def __init__(self, root, capacity=4096):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        self.count=0
        self.capacity=capacity
        self.arrays={}
        for key,shape,dtype in [("raw_state",(capacity+1,61),np.float32),
            ("actions",(capacity,23),np.float32),("rewards",(capacity,),np.float32),
            ("dones",(capacity,),bool),("truncated",(capacity,),bool)]:
            self.arrays[key]=np.lib.format.open_memmap(self.root/(key+".npy"),mode="w+",dtype=dtype,shape=shape)
        for i in range(3):
            self.arrays[f"image_{i}"]=np.lib.format.open_memmap(self.root/f"image_{i}.npy",mode="w+",dtype=np.uint8,
                shape=(capacity+1,224,224,3))

    def observation(self, obs, index):
        self.arrays["raw_state"][index]=array(obs["robot_r1::proprio"])
        for i,camera in enumerate(("zed_link","left_realsense_link","right_realsense_link")):
            image=array(obs[f"robot_r1::robot_r1:{camera}:Camera:0::rgb"])[...,:3]
            self.arrays[f"image_{i}"][index]=cv2.resize(image,(224,224),interpolation=cv2.INTER_LINEAR)

    def before_step(self, obs):
        if self.count>=self.capacity:
            raise ValueError("Online replay capacity exceeded")
        self.observation(obs,self.count)

    def after_step(self, action,reward,terminated,truncated,obs):
        self.arrays["actions"][self.count]=array(action)
        self.arrays["rewards"][self.count]=float(reward)
        self.arrays["dones"][self.count]=bool(terminated)
        self.arrays["truncated"][self.count]=bool(truncated)
        self.observation(obs,self.count+1)
        self.count+=1
        if self.count%64==0 or terminated or truncated:
            for data in self.arrays.values():data.flush()
            report={"transitions":self.count,"observations":self.count+1,
                "complete":bool(terminated or truncated),"terminated":bool(terminated),"truncated":bool(truncated),
                "terminal_reward":float(reward),"reward_source":"actual env.step reward",
                "capacity":self.capacity,"note":"Read only manifest transitions; unused preallocated rows are not data."}
            temporary=self.root/"manifest.tmp"
            temporary.write_text(json.dumps(report,indent=2)+"\n")
            temporary.replace(self.root/"manifest.json")

