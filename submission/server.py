"""Policy-only single-client msgpack websocket entrypoint, compatible with stock evaluator."""
import argparse
import asyncio
import functools
import http
import json
import logging
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import websockets
import websockets.asyncio.server
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from submission.contract import TaskRouter,TASKS
from adapters.b1k_websocket_server import Packer,unpackb

class Server:
    def __init__(self,policy,metadata):self.policy=policy;self.metadata=metadata;self.client=False
    async def health(self,conn,request):
        if request.path=='/healthz':return conn.respond(http.HTTPStatus.OK,'OK\n')
        if request.path=='/metadata':return conn.respond(http.HTTPStatus.OK,json.dumps(self.metadata))
    async def handle(self,ws):
        if self.client:
            await ws.close(code=1013,reason='One evaluator connection per worker');return
        self.client=True
        try:
            self.policy.reset();packer=Packer();await ws.send(packer.pack(self.metadata))
            async for data in ws:
                obs=unpackb(data,strict_map_key=False)
                if 'reset' in obs:self.policy.reset();continue # Official reset has no reply.
                if '__action_chunk_size__' in obs:raise ValueError('Action-chunk replay is disabled')
                tick=time.monotonic();action=self.policy.act(obs)
                await ws.send(packer.pack({'action':action,'server_timing':{'infer_ms':(time.monotonic()-tick)*1000}}))
        except Exception:
            logging.error(traceback.format_exc());await ws.close(code=1011,reason='Policy inference failure')
        finally:
            self.policy.reset();self.client=False
    async def run(self,host,port):
        async with websockets.asyncio.server.serve(self.handle,host,port,compression=None,max_size=None,process_request=self.health,ping_interval=60,ping_timeout=300):
            await asyncio.Future()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model-root',type=Path,default=Path('/models'));p.add_argument('--log-root',type=Path,default=Path('/logs'));p.add_argument('--port',type=int,default=8000);p.add_argument('--host',default='0.0.0.0');p.add_argument('--seed',type=int,default=0);p.add_argument('--contract-smoke-only',action='store_true');a=p.parse_args()
    m=json.loads(a.manifest.read_text())
    if a.contract_smoke_only:
        class Mock:
            def act(self,obs):return np.zeros(23,np.float32)
            def reset(self):pass
        f=lambda task,route:Mock()
    else:
        from submission.policy import factory
        f=functools.partial(factory,model_root=a.model_root,log_root=a.log_root,seed=a.seed)
    router=TaskRouter(m['routes'],f)
    metadata={'supported_tasks':[TASKS[int(t)] for t in m['routes']],'batch_size':1,'action_shape':[1,23],
              'unsupported_task_behavior':'zero action; outside declared submission scope',
              'contract_smoke_only':a.contract_smoke_only,'policy_review_status':m['status'],'action_chunk_replay':False}
    try:asyncio.run(Server(router,metadata).run(a.host,a.port))
    finally:router.close()
if __name__=='__main__':main()
