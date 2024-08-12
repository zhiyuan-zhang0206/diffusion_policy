from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
import lightning.pytorch as L
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_location_rotation_separate_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)
                replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
    
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
        
        # aggregate obs stats
        obs_stat = array_to_stats(self.replay_buffer['obs'])


        normalizer['obs'] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

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
    
    data = {
        'obs': obs,
        'action': raw_actions
    }
    return data

class RobomimicLowdimDatamodule(L.LightningDataModule):
    def __init__(self,dataset: RobomimicReplayLowdimDataset, batch_size:int=128, num_workers:int=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

        self.train_dataset = self.dataset
        self.val_dataset = self.train_dataset.get_validation_dataset()
        self.normalizer = self.train_dataset.get_normalizer()

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True , persistent_workers=True if self.num_workers!=0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True , persistent_workers=True if self.num_workers!=0 else False)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True if self.num_workers!=0 else False)
    

class MixedRobomimicReplayLowdimDataset:
    def __init__(self, datasets: List[RobomimicReplayLowdimDataset]):
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
        return MixedRobomimicReplayLowdimDataset([dataset.get_validation_dataset() for dataset in self.datasets])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_index = self.data_index_to_dataset_index[idx]
        normalizer = self.normalizers[dataset_index]
        data = self.datasets[int(dataset_index)][idx - self.dataset_start_indices[dataset_index]]
        data['normalized_action'] = normalizer['action'].normalize(data['action'])
        return data
    
    def get_normalizer(self):
        return LinearNormalizer()
    
class MixedRobomimicLowdimDatamodule(L.LightningDataModule):
    def __init__(self,dataset: MixedRobomimicReplayLowdimDataset, batch_size:int=128, num_workers:int=0):
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
    