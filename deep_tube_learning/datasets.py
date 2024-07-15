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
        z_e = epoch_data['z']
        v_e = epoch_data['v']
        done_e = epoch_data['done']
        done_e[-1, :] = True
        pz_x_e = epoch_data['pz_x']

        # Place the 'robots' axis first, 'time' axis second
        # TODO: change this so that z is dataset x time x state, where dataset = robots x epochs
        z_e = np.swapaxes(z_e, 0, 1)
        v_e = np.swapaxes(v_e, 0, 1)
        pz_x_e = np.swapaxes(pz_x_e, 0, 1)
        done_e = np.swapaxes(done_e, 0, 1)

        # Concatenate with other data
        if z is None:
            z = z_e
            v = v_e
            done = done_e
            pz_x = pz_x_e
        else:
            z = np.concatenate((z, z_e), axis=0)
            v = np.concatenate((v, v_e), axis=0)
            done = np.concatenate((done, done_e), axis=0)
            pz_x = np.concatenate((pz_x, pz_x_e), axis=0)

    # Translate arrays by one for next steps
    z_p1 = z[:, 1:, :].copy()
    pz_x_p1 = pz_x[:, 1:, :].copy()

    # Save dataset

    dataset = {
        'z': z,
        'pz_x': pz_x,
        'v': v,
        'z_p1': z_p1,
        'pz_x_p1': pz_x_p1,
        'done': done
    }

    with open(f"{data_folder}/dataset.pickle", "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def get_slice(data, i, dN, m):
    dc = data.copy()
    slc = np.flip(np.arange(dc.shape[-2] - (i * dN) - 1, -1, step=-dN))
    start = data[:, 0, :].reshape((data.shape[0], 1, data.shape[2]))
    start[:, :, :-m] = 0
    return np.concatenate((np.repeat(start, dc.shape[-2] - len(slc), axis=-2), dc[:, slc, :]), axis=-2)


def sliding_window(data, N, dN, m):
    return np.concatenate([get_slice(data, i, dN, m) for i in range(N)], axis=-1)


def get_dataset(wandb_experiment):
    data_folder = f"rom_tracking_data/{wandb_experiment}"
    dataset_file = f"{data_folder}/dataset.pickle"
    if not os.path.isfile(dataset_file):
        if not os.path.isfile(f"{data_folder}/epoch_0.pickle"):
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
    def from_wandb(cls, wandb_experiment, N=1, dN=1):
        dataset = get_dataset(wandb_experiment)

        z = dataset['z'][:, :-1, :]
        pz_x = dataset['pz_x'][:, :-1, :]

        # Compute error terms
        w = np.linalg.norm(pz_x - z, axis=-1)
        w_p1 = np.linalg.norm(dataset['pz_x_p1'] - dataset['z_p1'], axis=-1)
        data = np.concatenate((w[:, :, None], z, dataset['v']), axis=-1)

        data = sliding_window(data, N, dN, dataset['v'].shape[-1])
        shp = data.shape
        data = data.reshape((shp[0] * shp[1], shp[2]))
        done = dataset['done'].reshape((shp[0] * shp[1],))
        w_p1 = w_p1.reshape((w_p1.shape[0] * w_p1.shape[1],))

        data = data[np.logical_not(done), :]
        w_p1 = w_p1[np.logical_not(done)]

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(w_p1[:, None]).float()
        input_dim = data.shape[1]
        output_dim = 1
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(ScalarTubeDataset, self).__init__(data, target, input_dim, output_dim)


class VectorTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment, N=1, dN=1):
        dataset = get_dataset(wandb_experiment)

        z = dataset['z'][:, :-1, :]
        pz_x = dataset['pz_x'][:, :-1, :]

        # Compute error terms
        w = np.abs(pz_x - z)
        w_p1 = np.abs(dataset['pz_x_p1'] - dataset['z_p1'])
        data = np.concatenate((w, z, dataset['v']), axis=-1)

        data = sliding_window(data, N, dN, dataset['v'].shape[-1])
        shp = data.shape
        data = data.reshape((shp[0] * shp[1], shp[2]))
        done = dataset['done'].reshape((shp[0] * shp[1],))
        w_p1 = w_p1.reshape((w_p1.shape[0] * w_p1.shape[1], w_p1.shape[2]))

        data = data[np.logical_not(done), :]
        w_p1 = w_p1[np.logical_not(done), :]

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(w_p1).float()
        input_dim = data.shape[1]
        output_dim = target.shape[1]
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(VectorTubeDataset, self).__init__(data, target, input_dim, output_dim)


class AlphaScalarTubeDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment, N=1, dN=1):
        dataset = get_dataset(wandb_experiment)

        z = dataset['z'][:, :-1, :]
        pz_x = dataset['pz_x'][:, :-1, :]

        # Compute error terms
        w = np.linalg.norm(pz_x - z, axis=-1)
        w_p1 = np.linalg.norm(dataset['pz_x_p1'] - dataset['z_p1'], axis=-1)
        data = np.concatenate((w[:, :, None], z, dataset['v']), axis=-1)

        data = sliding_window(data, N, dN, dataset['v'].shape[-1])
        shp = data.shape
        data = data.reshape((shp[0] * shp[1], shp[2]))
        done = dataset['done'].reshape((shp[0] * shp[1],))
        w_p1 = w_p1.reshape((w_p1.shape[0] * w_p1.shape[1],))

        data = data[np.logical_not(done), :]
        w_p1 = w_p1[np.logical_not(done)]

        alpha = np.random.uniform(size=(data.shape[0], 1))
        data = np.hstack((data, alpha))

        data = torch.from_numpy(data).float()
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
    def from_wandb(cls, wandb_experiment, N=1, dN=1):
        dataset = get_dataset(wandb_experiment)

        z = dataset['z'][:, :-1, :]
        pz_x = dataset['pz_x'][:, :-1, :]

        # Compute error terms
        w = np.abs(pz_x - z)
        w_p1 = np.abs(dataset['pz_x_p1'] - dataset['z_p1'])
        data = np.concatenate((w, z, dataset['v']), axis=-1)

        data = sliding_window(data, N, dN, dataset['v'].shape[-1])
        shp = data.shape
        data = data.reshape((shp[0] * shp[1], shp[2]))
        done = dataset['done'].reshape((shp[0] * shp[1],))
        w_p1 = w_p1.reshape((w_p1.shape[0] * w_p1.shape[1], w_p1.shape[2]))

        data = data[np.logical_not(done), :]
        w_p1 = w_p1[np.logical_not(done), :]

        alpha = np.random.uniform(size=(data.shape[0], 1))
        data = np.hstack((data, alpha))

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(w_p1).float()
        input_dim = data.shape[1]
        output_dim = target.shape[1]
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(AlphaVectorTubeDataset, self).__init__(data, target, input_dim, output_dim)

    def update(self):
        self.data[:, -1] = torch.rand(size=(len(self),))


class ErrorDynamicsDataset(TubeDataset):

    @classmethod
    def from_wandb(cls, wandb_experiment, N=1, dN=1):
        dataset = get_dataset(wandb_experiment)

        z = dataset['z'][:, :-1, :]
        pz_x = dataset['pz_x'][:, :-1, :]

        # Compute error terms
        w = pz_x - z
        w_p1 = dataset['pz_x_p1'] - dataset['z_p1']
        data = np.concatenate((w, z, dataset['v']), axis=-1)

        data = sliding_window(data, N, dN, dataset['v'].shape[-1])
        shp = data.shape
        data = data.reshape((shp[0] * shp[1], shp[2]))
        done = dataset['done'].reshape((shp[0] * shp[1],))
        w_p1 = w_p1.reshape((w_p1.shape[0] * w_p1.shape[1], w_p1.shape[2]))

        data = data[np.logical_not(done), :]
        w_p1 = w_p1[np.logical_not(done), :]

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(w_p1).float()
        input_dim = data.shape[1]
        output_dim = target.shape[1]
        return cls(data, target, input_dim, output_dim)

    def __init__(self, data, target, input_dim, output_dim):
        super(ErrorDynamicsDataset, self).__init__(data, target, input_dim, output_dim)