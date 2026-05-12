"""
RoboMemArena HDF5 to RLDS (TFDS) Dataset Conversion Script

Converts RoboMemArena demonstration data (HDF5 format) to RLDS format for training.
Data structure: each task has full_trajectory/ and subtask_data/ folders;
this script reads from subtask_data/*.hdf5 which contain keyframe-annotated episodes.
"""
import os
import glob

os.environ['NO_GCE_CHECK'] = 'true'
try:
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
except ImportError:
    pass

from typing import Iterator, Tuple, Any
import h5py
import numpy as np
import tensorflow_datasets as tfds
from conversion_utils import MultiThreadedDatasetBuilder

# Data root: adjust for your environment.
DATA_ROOT = os.environ.get("ROBOMEMARENA_DATA_ROOT", "/path/to/robomemarena_data")


def _collect_hdf5_paths():
    """Collect all HDF5 files from 26 task subtask_data folders."""
    paths = []
    pattern = os.path.join(DATA_ROOT, "*_dataset", "subtask_data", "*.hdf5")
    for p in glob.glob(pattern):
        paths.append(p)
    return paths


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    def _parse_example(episode_path, demo_id):
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

        raw_file_string = os.path.basename(episode_path)
        name = raw_file_string[:-5]  # remove .hdf5
        parts = name.split("_")
        instr_words = parts[:-3]  # remove variant, seed, task suffix
        if instr_words and instr_words[-1].endswith('.'):
            instr_words[-1] = instr_words[-1][:-1]
        instruction = " ".join(instr_words)

        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i],
                    'wrist_image': wrist_images[i],
                    'state': np.asarray(
                        np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': instruction,
            })

        sample = {
            'steps': episode,
            'episode_metadata': {'file_path': episode_path}
        }
        return episode_path + f"_{demo_id}", sample

    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])
        idx = 0
        cnt = 0
        while cnt < n_demos:
            ret = _parse_example(sample, idx)
            if ret is not None:
                cnt += 1
            idx += 1
            yield ret


class RoboMemArenaDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for RoboMemArena 26-task dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    N_WORKERS = 4
    MAX_PATHS_IN_MEMORY = 8
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format='jpeg'),
                        'wrist_image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format='jpeg'),
                        'state': tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        'joint_state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        paths = _collect_hdf5_paths()
        return {'train': paths}
