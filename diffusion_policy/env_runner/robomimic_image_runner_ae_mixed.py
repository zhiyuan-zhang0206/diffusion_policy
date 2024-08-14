import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import pickle
from pathlib import Path
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicLowdimDatamodule
from diffusion_policy.dataset.robomimic_language_description import get_language_description, get_language_embedding
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from loguru import logger


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunnerAEMixed:
    """
    Robomimic envs already enforces number of steps.
    This is for testing autoencoders only.
    """

    def __init__(self, 
            output_dir,
            shape_meta: dict = {
                'obs': {
                    'agentview_image': {
                        'shape': [3, 84, 84],
                        'type': 'rgb'
                    },
                    'robot0_eef_pos': {
                        'shape': [3]
                        # type default: low_dim
                    },
                    'robot0_eef_quat': {
                        'shape': [4]
                    },
                    'robot0_gripper_qpos': {
                        'shape': [2]
                    }
                },
                'action': {
                    'shape': [10]
                }
            },
            obs_keys:list=[],
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=196,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=2,
            render_hw=[128,128],
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            runners=None,
        ):
        self.runners = runners

    def set_normalizer(self, workspace):
        for runner, dataset in zip(self.runners, workspace.datamodule.dataset.datasets):
            runner.normalizer = dataset.get_normalizer()

    def load_replay_buffer(self, workspace):
        for runner, dataset in zip(self.runners, workspace.datamodule.dataset.datasets):
            runner.replay_buffer = dataset.replay_buffer
        self.datamodule: RobomimicLowdimDatamodule = workspace.datamodule

    def set_action_steps(self, n_action_steps):
        if n_action_steps is not None:
            for runner in self.runners:
                runner.set_action_steps(n_action_steps)

    def run(self, policy: BaseImagePolicy, verbose=False):
        results = []
        for runner in self.runners:
            result = runner.run(policy, verbose)
            result['task_name'] = runner.task_name
            results.append(result)
        return results

    def close(self):
        for runner in self.runners:
            runner.close()
        return

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

def compute_train_mean_score(log_data):
    scores = []
    for key, value in log_data.items():
        if 'train' in key and 'max_reward' in key:
            scores.append(value)
    if len(scores) == 0:
        return 
    else:
        return np.mean(scores)
    
def format_log_data(log_data):
    # exam train video that's not successful
    formatted = {}
    for key, value in log_data.items():
        if 'train' in key and 'max_reward' in key:
            if value < 0.9:
                formatted[key] = value
    return formatted