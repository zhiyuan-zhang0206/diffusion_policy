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
import wandb.sdk.data_types.video as wv
from PIL import Image
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import zzy_utils
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


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=0,
            n_train_vis=3,
            train_start_idx=0,
            n_test=25,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=1,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if zzy_utils.check_environ_debug():
            max_steps = 56
        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False
        # env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = None
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
                    if zzy_utils.check_environ_debug():
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', zzy_utils.get_current_run_timestamp() + ".mp4")
                        logger.info(f"Saving video to {filename}")
                    else:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
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

        ############################################
        self.image_keys = sorted([render_obs_key, 'robot0_eye_in_hand_image'])
        self.low_dim_keys = sorted(['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])

    def run(self, policy, language_feature, img_feature_extract_fn, img_preprocess):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
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

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            if zzy_utils.check_environ_debug():
                from pathlib import Path
                import pickle
                with Path(__file__).parent.joinpath('env_actions.pkl').open('rb') as f:
                    self.env_actions : list = pickle.load(f)['qpos'].tolist()
                
                # env_actions_array = np.concatenate([arr[0] for arr in self.env_actions])['qpos']
                # env_actions_array[:, 0] =  0.27
                # env_actions_array[:, 2] =  1.0
                # env_actions_array[:, 3:6] = np.array([0,  0, 0,])
                # env_actions_array[25:50, 3:6] = np.array([1.57, 0,  0,])
                # env_actions_array[50:75, 3:6] = np.array([0, 1.57, 0,])
                # env_actions_array[75:100, 3:6] = np.array([0,  0,1.57,])
                # env_actions_array[:, -1] = -1
                # for line in env_actions_array:
                #     print(' '.join([f'{val:.4f}' for val in line]))
                # self.env_actions = [env_actions_array[i*8:(i+1)*8][np.newaxis] for i in range(len(env_actions_array)//8)]


            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                
                views = [np_obs_dict[k] for k in sorted(self.image_keys)]
                low_dim_data = [np_obs_dict[k] for k in sorted(self.low_dim_keys)]
                batch_size = n_envs

                batch_image_tensors = []
                for i in range(batch_size):
                    view_0 = (np.squeeze(views[0][i]) * 255).astype(np.uint8).transpose(1,2,0)
                    view_1 = (np.squeeze(views[1][i]) * 255).astype(np.uint8).transpose(1,2,0)
                    processed_0 = img_preprocess(Image.fromarray(view_0))
                    processed_1 = img_preprocess(Image.fromarray(view_1))
                    batch_image_tensors.append(torch.stack([processed_0, processed_1], dim=0))
                batch_image_tensors = torch.stack(batch_image_tensors, dim=0).view(batch_size * 2, 3, 224, 224).to(device=device, dtype=dtype)
                image_features = img_feature_extract_fn(batch_image_tensors).view(batch_size, 2, -1)
                low_dim_data = np.squeeze(np.concatenate(low_dim_data, axis=-1))
                low_dim_data = torch.from_numpy(low_dim_data).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
                language_feature = language_feature.expand(batch_size, 1, -1)
                # run policy
                with torch.no_grad():
                    action = policy.predict_action(raw_language_features=language_feature, raw_image_features=image_features, raw_low_dim_data=low_dim_data).cpu().numpy()

                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                pred_action_length = action.shape[1]

                if self.n_action_steps > pred_action_length // 2:
                        env_action = action[:, :self.n_action_steps, :]
                else:
                    # step env
                    if past_action is None:
                        env_action = action[:, :self.n_action_steps, :]
                    else:
                        env_action = (action[:, :self.n_action_steps, :] + past_action[:, self.n_action_steps: self.n_action_steps*2:, :]) / 2

                if zzy_utils.check_environ_debug():
                    env_action = self.env_actions[:8]
                    self.env_actions = self.env_actions[8:]
                    env_action = np.array(env_action)[None, ...]
                    # env_action = np.diff(env_action, axis=1)
                    # logger.info(env_action.shape)
                    # env_action[0, :, :] = 0.0
                    # env_action[0, :, -1] = -1
                    # env_action[0, :, 0] = 0.05
                    # env_action[0, :, 1] = 0.05
                    # env_action[0, :, 2] = 0.05
                    pass
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(env_action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        env.close()
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
            # if video_path is not None:
            #     sim_video = wandb.Video(video_path)
            #     log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

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
