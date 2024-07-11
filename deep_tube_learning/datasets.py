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

    def __init__(self, data, target, input_dim, output_dim):
        self.data = data
        self.target = target
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.target[idx, :]

    def update(self):
        pass

    def random_split(self, split_proportion):
        """
        Splits the dataset into two random contiguous pieces
        :param split_proportion:
        :return:
        """
        split_len = int(len(self) * split_proportion)
        idx = np.random.randint(len(self) - split_len)

        data1 = self.data[idx:split_len + idx]
        data2 = torch.vstack((self.data[:idx], self.data[split_len + idx:]))
        target1 = self.target[idx:split_len + idx]
        target2 = torch.vstack((self.target[:idx], self.target[split_len + idx:]))

        return type(self)(data1, target1, self.input_dim, self.output_dim), type(self)(data2, target2, self.input_dim, self.output_dim)


class ScalarTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment):
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        # TODO: is just applying the norm here correct?
        w = np.linalg.norm(dataset['z'] - dataset['pz_x'], axis=1)
        w_p1 = np.linalg.norm(dataset['z_p1'] - dataset['pz_x_p1'], axis=1)
        data = torch.from_numpy(np.hstack((w[:, None], dataset['z'], dataset['v']))).float()
        target = torch.from_numpy(w_p1[:, None]).float()

        input_dim = data.shape[1]
        output_dim = 1
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(ScalarTubeDataset, self).__init__(data, target, input_dim, output_dim)


class VectorTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment):
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        w = np.abs(dataset['z'] - dataset['pz_x'])
        w_p1 = np.abs(dataset['z_p1'] - dataset['pz_x_p1'])
        data = torch.from_numpy(np.hstack((w, dataset['z'], dataset['v']))).float()
        target = torch.from_numpy(w_p1).float()

        input_dim = data.shape[1]
        output_dim = target.shape[1]
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(VectorTubeDataset, self).__init__(data, target, input_dim, output_dim)


class AlphaScalarTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment):
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        # TODO: is just applying the norm here correct?
        w = np.linalg.norm(dataset['z'] - dataset['pz_x'], axis=1)
        w_p1 = np.linalg.norm(dataset['z_p1'] - dataset['pz_x_p1'], axis=1)
        alpha = np.random.uniform(size=(dataset.shape[0], 1))
        data = torch.from_numpy(np.hstack((w[:, None], dataset['z'], dataset['v'], alpha))).float()
        target = torch.from_numpy(w_p1[:, None]).float()

        input_dim = data.shape[1]
        output_dim = 1
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(AlphaScalarTubeDataset, self).__init__(data, target, input_dim, output_dim)

    def update(self):
        self.data[:, -1] = torch.rand(size=(len(self),))


class AlphaVectorTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment):
        dataset = get_dataset(wandb_experiment)

        # Compute error terms
        w = np.abs(dataset['z'] - dataset['pz_x'])
        w_p1 = np.abs(dataset['z_p1'] - dataset['pz_x_p1'])
        alpha = np.random.uniform(size=(dataset.shape[0], 1))
        data = torch.from_numpy(np.hstack((w, dataset['z'], dataset['v'], alpha))).float()
        target = torch.from_numpy(w_p1).float()

        input_dim = data.shape[1]
        output_dim = target.shape[1]
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(AlphaVectorTubeDataset, self).__init__(data, target, input_dim, output_dim)

    def update(self):
        self.data[:, -1] = torch.rand(size=(len(self),))
