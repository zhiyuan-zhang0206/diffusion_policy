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


class RobomimicImageRunnerAE(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    This is for testing autoencoders only.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
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
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        self.task_name = '_'.join(dataset_path.split('/')[-3:-1])
        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        self.language_description = get_language_description(dataset_path)
        self.language_embedding = get_language_embedding(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', str(i) + '_train_' + wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', str(i) + '_test_' + wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = n_train_vis
        self.n_test_vis = n_test_vis
        self.n_envs = n_envs
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def load_replay_buffer(self, workspace):
        self.replay_buffer: ReplayBuffer = workspace.datamodule.dataset.replay_buffer
        self.datamodule: RobomimicLowdimDatamodule = workspace.datamodule

    def set_action_steps(self, n_action_steps):
        if n_action_steps is not None:
            self.n_action_steps = int(n_action_steps)

    def run(self, policy: BaseImagePolicy, verbose=False):
        device = policy.device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = policy.dtype
        env = self.env
        self.horizon = policy.pl_model.horizon
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # dataloader = self.datamodule.predict_dataloader()
        policy.pl_model.eval()
        # model = DownsampleCVAE.load_from_checkpoint('/home/zzy/robot/robot_zzy/diffusion_policy/lightning_logs/r8ik38j3/checkpoints/last.ckpt')
        # for batch in dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     loss_dict, pred_action = policy.pl_model(batch)
        #     print(loss_dict)
        #     break
        # return
        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            # logger.debug(f'chunk_idx: {chunk_idx}')
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            if hasattr(self, 'replay_buffer'):
                chunk_actions = []
                for i in range(self.n_envs):
                    if chunk_idx == 0 and i == 0:
                        start = 0
                    else:
                        start = self.replay_buffer.episode_ends[chunk_idx * self.n_envs + i - 1]
                    end = self.replay_buffer.episode_ends[chunk_idx * self.n_envs + i]
                    actions = self.replay_buffer['action'][start:end]
                    tail = [actions[-1]] * (self.horizon * 3 - 1)
                    actions = np.concatenate([actions, tail])
                    chunk_actions.append(actions)
                max_length = max( max([len(actions) for actions in chunk_actions]) + 1, self.max_steps)
                chunk_actions = [np.concatenate([actions, np.array([actions[-1]] * (max_length - len(actions)))]) for actions in chunk_actions]
                chunk_actions = np.stack(chunk_actions)

                self.env_actions = [chunk_actions[:, i * self.n_action_steps : i * self.n_action_steps +self.horizon] for i in range(len(chunk_actions[0])//self.n_action_steps)]
            else:
                self.env_actions = [np.zeros((1, 10))] * 1000

            done = False
            index = -1
            while not done:
                index += 1
                # logger.debug(f'index: {index}')
                # create obs dict
                obs['language_embedding'] = self.language_embedding
                np_obs_dict = dict(obs)
                # if self.past_action and (past_action is not None):
                #     # TODO: not tested
                #     np_obs_dict['past_action'] = past_action[
                #         :,-(self.n_obs_steps-1):].astype(np.float32)
                
                data_action = self.env_actions[index]
                np_obs_dict['action'] = data_action
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy(), True)
                
                action = np_action_dict['action'][:, :self.n_action_steps, :]

                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        _ = env.reset()
        # env.close()
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        # logger.info(log_data)
        if verbose:
            logger.info(f'train mean score: {compute_train_mean_score(log_data)}')
            logger.info(f'failed train: {format_log_data(log_data)}')
            if "test/mean_score" in log_data:
                logger.info(f'test mean score: {log_data["test/mean_score"]}')
            else:
                logger.info(f'test mean score: None')
        return log_data

    def close(self):
        self.env.close()
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