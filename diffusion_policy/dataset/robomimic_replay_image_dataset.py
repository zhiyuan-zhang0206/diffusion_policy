if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))



from typing import Dict, List, Literal
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
from pathlib import Path
import concurrent.futures
import multiprocessing
from torch.utils.data import DataLoader
import lightning as L
from omegaconf import OmegaConf
from torch.utils.data import Dataset, ConcatDataset
import hydra
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.robomimic_language_description import get_language_description, get_language_embedding
from typing import Union
import pickle
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_location_rotation_separate_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

def _get_actions(dataset_path: str):
    action_path = Path(Path(dataset_path).as_posix().removesuffix('.hdf5') + '_actions.npy')
    if not action_path.exists():
        raise FileNotFoundError(f'Actions file not found at {action_path}')
    with action_path.open('rb') as f:
        data = np.load(f)
    return data

def _get_image_features(dataset_path: str):
    image_feature_path = Path(Path(dataset_path).as_posix().removesuffix('.hdf5') + '_features.npy')
    if not image_feature_path.exists():
        raise FileNotFoundError(f'Image features file not found at {image_feature_path}')
    with image_feature_path.open('rb') as f:
        data = np.load(f)
    return data

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)
        self.val_ratio = val_ratio
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_path = dataset_path

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                # this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
                this_normalizer = robomimic_abs_action_only_location_rotation_separate_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

class RobomimicReplayImageLanguageDataset(RobomimicReplayImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            load_image_features=False
        ):
        """
        If load_image_features is True, load R3M features directly, ignoring images.
        """
        super().__init__(shape_meta, dataset_path, horizon, pad_before, pad_after, n_obs_steps, abs_action, rotation_rep, use_legacy_normalizer, use_cache, seed, val_ratio)
        # self.language_description = get_language_description(dataset_path)
        self.language_embedding = get_language_embedding(dataset_path)
        self.language_description = get_language_description(dataset_path)
        self.load_image_features = load_image_features
        if self.load_image_features:
            self.image_features = _get_image_features(dataset_path)
            self.actions = _get_actions(dataset_path)
            assert len(self.image_features) == len(self.actions)
            length = super().__len__() + super().get_validation_dataset().__len__()
            assert len(self.actions) == length, f"The length of actions and the dataset should be the same: {len(self.actions)} vs {length}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.load_image_features:
            data = {'obs':{}}
            data['obs']['image_features'] = torch.from_numpy(self.image_features[idx])
            data['action'] = torch.from_numpy(self.actions[idx])
        else:
            data = super().__getitem__(idx)

        data['obs']['language_description'] = self.language_description
        data['obs']['language_embedding'] = self.language_embedding
        if not self.load_image_features:
            data['obs']['image'] = data['obs']['agentview_image']
            del data['obs']['agentview_image']
        return data
    
class MixedRobomimicReplayImageDataset:
    def __init__(self, datasets: List[RobomimicReplayImageLanguageDataset]):
        self.datasets = datasets
        self.normalizers = [dataset.get_normalizer() for dataset in datasets]
        data_index_to_dataset_index = []
        for i, dataset in enumerate(datasets):
            data_index_to_dataset_index.extend([i] * len(dataset))
        self.data_index_to_dataset_index = np.array(data_index_to_dataset_index, dtype=np.int32)
        
        dataset_lengths = np.array([len(dataset) for dataset in datasets])
        self.dataset_start_indices = [0] + np.cumsum(dataset_lengths).tolist()[:-1]
        
        self.length = sum([len(dataset) for dataset in datasets])

    def get_validation_dataset(self):
        return MixedRobomimicReplayImageDataset([dataset.get_validation_dataset() for dataset in self.datasets])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_index = self.data_index_to_dataset_index[idx]
        normalizer = self.normalizers[dataset_index]
        data = self.datasets[int(dataset_index)][idx - self.dataset_start_indices[dataset_index]]
        data['normalized_action'] = normalizer['action'].normalize(data['action']).detach()
        return data
    
    def get_normalizer(self):
        return LinearNormalizer()

class RobomimicImageDatamodule(L.LightningDataModule):
    def __init__(self, dataset: Union[RobomimicReplayImageDataset, RobomimicReplayImageLanguageDataset],
                 batch_size:int=128, num_workers:int=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

        if self.dataset.load_image_features:
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * self.dataset.val_ratio)
            train_size = dataset_size - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, 
                [train_size, val_size],
            )
        else:
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset.get_validation_dataset()
        self.normalizer = self.dataset.get_normalizer()

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(ConcatDataset([self.train_dataset, self.val_dataset]), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        

class MixedRobomimicImageDatamodule(L.LightningDataModule):
    def __init__(self,dataset: MixedRobomimicReplayImageDataset, batch_size:int=128, num_workers:int=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

        self.train_dataset = self.dataset
        self.val_dataset = self.train_dataset.get_validation_dataset()

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True , persistent_workers=True if self.num_workers!=0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False , persistent_workers=False)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    

def main():
    dataset = MixedRobomimicReplayImageLanguageDataset(
                "/home/zzy/robot/data/diffusion_policy_data/data/robomimic/datasets",
                include_tasks=['lift', 'can', 'square'],
                data_type='ph',
                shape_meta= {
                    'obs': {
                        'agentview_image': {
                            'shape': [3, 84, 84],
                            'type': 'rgb'
                            },
                            'robot0_eye_in_hand_image': {
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
                horizon=16,
                pad_before=0,
                pad_after=15,
                n_obs_steps=None,
                abs_action=True,
                rotation_rep='rotation_6d',
                use_legacy_normalizer=False,
                use_cache=False,
                seed=42,
                val_ratio=0.05
                )





if __name__ == '__main__':
    main()