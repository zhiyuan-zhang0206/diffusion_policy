"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import zzy_utils
import signal
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os
import signal
import psutil

def terminate_child_processes():
    print("terminating child processes")
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    
    gone, alive = psutil.wait_procs(children, timeout=3)
    
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    print("terminated all child processes. They are gone.")

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--n_action_steps', default=None)
def main(checkpoint, output_dir, device, n_action_steps):
    if os.path.exists(output_dir) and not zzy_utils.check_environ_debug():
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    for c in cfg.task.env_runner.runners:
        c.n_envs = 1 
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if hasattr(cfg, 'training') and hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    if zzy_utils.check_environ_debug():
        if hasattr(cfg, 'test_ae_only') and cfg.test_ae_only:
            cfg.task.env_runner['n_test'] = 0
            cfg.task.env_runner['n_train'] = 50
        else:
            cfg.task.env_runner['n_test'] = 50
            cfg.task.env_runner['n_train'] = 50
            
        cfg.task.env_runner['n_test_vis'] = cfg.task.env_runner['n_test']
        cfg.task.env_runner['n_train_vis'] = cfg.task.env_runner['n_train']
        cfg.task.env_runner['n_envs'] =  min(max(cfg.task.env_runner['n_train'], cfg.task.env_runner['n_test']), 10)
        cfg.task.env_runner['max_steps'] = 128

    env_runner = workspace.env_runner
    # env_runner = hydra.utils.instantiate(
    #     cfg.task.env_runner,
    #     output_dir=output_dir)
    if hasattr(cfg, 'test_ae_only') and cfg.test_ae_only:
        env_runner.load_replay_buffer(workspace)
    env_runner.set_action_steps(n_action_steps)
    runner_log = env_runner.run(policy, verbose=True)
    if hasattr(env_runner, 'close'):
        env_runner.close()
    print(runner_log)
    # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    terminate_child_processes()

if __name__ == '__main__':
    main()


