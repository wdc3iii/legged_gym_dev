import os
import glob
import wandb
import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def construct_dataset(data_folder):
    epoch_files = glob.glob(f"{data_folder}/epoch_*.pickle")
    z, v, pz_x, done, data_shape = None, None, None, None, None
    for epoch_file in epoch_files:
        with open(epoch_file, "rb") as f:
            epoch_data = pickle.load(f)

        # Reshape data to remove the 'num_robots' axis
        # Note that by removing N+1 element from z, Pz_x, all arrays now have same leading axis
        # Enforce last element of leading axis always has done=True, so
        # removing 'num_robots' axis will not lead to spurious transition
        z_e = epoch_data['z'][:-1, :, :]
        data_shape = z_e.shape
        z_e = z_e.reshape((data_shape[0] * data_shape[1], -1), order='F')
        v_e = epoch_data['v'].reshape((data_shape[0] * data_shape[1], -1), order='F')
        done_e = epoch_data['done']
        done_e[-1, :] = True
        done_e = done_e.reshape((data_shape[0] * data_shape[1], -1), order='F')
        pz_x_e = epoch_data['pz_x'][:-1, :, :].reshape((data_shape[0] * data_shape[1], -1), order='F')

        # Concatenate with other data
        if z is None:
            z = z_e
            v = v_e
            done = done_e
            pz_x = pz_x_e
        else:
            z = np.vstack((z, z_e))
            v = np.vstack((v, v_e))
            done = np.vstack((done, done_e))
            pz_x = np.vstack((pz_x, pz_x_e))

    # Translate arrays by one for next steps
    z_p1 = z
    z_p1[:-1, :] = z[1:, :]
    pz_x_p1 = pz_x
    pz_x_p1[:-1, :] = pz_x[1:, :]

    # Remove 'done' transitions
    done = np.squeeze(done)
    z = z[np.logical_not(done), :]
    v = v[np.logical_not(done), :]
    pz_x = pz_x[np.logical_not(done), :]
    z_p1 = z_p1[np.logical_not(done), :]
    pz_x_p1 = pz_x_p1[np.logical_not(done), :]

    # Save dataset
    dataset = {
        'z': z,
        'pz_x': pz_x,
        'v': v,
        'z_p1': z_p1,
        'pz_x_p1': pz_x_p1,
    }

    with open(f"{data_folder}/dataset.pickle", "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def get_dataset(wandb_experiment):
    data_folder = f"rom_tracking_data/{wandb_experiment}"
    dataset_file = f"{data_folder}/dataset.pickle"
    if not os.path.isfile(dataset_file):
        experiment = f"coleonguard-Georgia Institute of Technology/RoM_Tracking_Data/{wandb_experiment}:latest"
        api = wandb.Api()
        artifact = api.artifact(experiment)
        artifact.download(root=Path(data_folder))
        dataset = construct_dataset(data_folder)
    else:
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)

    return dataset


class TubeDataset(Dataset):

    def __init__(self):
        self.data = None
        self.target = None
        self.input_dim = None
        self.output_dim = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.target[idx, :]

    def update(self):
        pass


class ScalarTubeDataset(TubeDataset):

    def __init__(self, wandb_experiment):
        super(ScalarTubeDataset, self).__init__()
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        # TODO: is just applying the norm here correct?
        w = np.linalg.norm(dataset['z'] - dataset['pz_x'], axis=1)
        w_p1 = np.linalg.norm(dataset['z_p1'] - dataset['pz_x_p1'], axis=1)
        self.data = torch.from_numpy(np.hstack((w[:, None], dataset['z'], dataset['v']))).float()
        self.target = torch.from_numpy(w_p1[:, None]).float()

        self.input_dim = self.data.shape[1]
        self.output_dim = 1


class VectorTubeDataset(TubeDataset):

    def __init__(self, wandb_experiment):
        super(VectorTubeDataset, self).__init__()
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        w = np.abs(dataset['z'] - dataset['pz_x'])
        w_p1 = np.abs(dataset['z_p1'] - dataset['pz_x_p1'])
        self.data = torch.from_numpy(np.hstack((w, dataset['z'], dataset['v']))).float()
        self.target = torch.from_numpy(w_p1).float()

        self.input_dim = self.data.shape[1]
        self.output_dim = self.target.shape[1]


class AlphaScalarTubeDataset(TubeDataset):

    def __init__(self, wandb_experiment):
        super(AlphaScalarTubeDataset, self).__init__()
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        # TODO: is just applying the norm here correct?
        w = np.linalg.norm(dataset['z'] - dataset['pz_x'], axis=1)
        w_p1 = np.linalg.norm(dataset['z_p1'] - dataset['pz_x_p1'], axis=1)
        alpha = np.random.uniform(size=(self.data[0], 1))
        self.data = torch.from_numpy(np.hstack((w[:, None], dataset['z'], dataset['v'], alpha))).float()
        self.target = torch.from_numpy(w_p1[:, None]).float()

        self.input_dim = self.data.shape[1]
        self.output_dim = 1

    def update(self):
        self.data[:, -1] = torch.rand(size=(len(self),))


class AlphaVectorTubeDataset(TubeDataset):

    def __init__(self, wandb_experiment):
        super(AlphaVectorTubeDataset, self).__init__()
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        w = np.abs(dataset['z'] - dataset['pz_x'])
        w_p1 = np.abs(dataset['z_p1'] - dataset['pz_x_p1'])
        alpha = np.random.uniform(size=(self.data[0], 1))
        self.data = torch.from_numpy(np.hstack((w, dataset['z'], dataset['v'], alpha))).float()
        self.target = torch.from_numpy(w_p1).float()

        self.input_dim = self.data.shape[1]
        self.output_dim = self.target.shape[1]

    def update(self):
        self.data[:, -1] = torch.rand(size=(len(self),))
