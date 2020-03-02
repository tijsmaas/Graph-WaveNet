import argparse
import pickle
from shutil import copyfile

import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean




def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj



def alter_dataset_precompute_scaler(dataset_dir, n_obs=None, fill_zeroes=False):
    """
    Alters dataset by adding the precomputed scaler, for faster bootstrapping.
    """
    print('Loading sensor values once to compute the scaler...')
    data = {}
    for category in ['train', 'val']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data[category] = cat_data
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std(), fill_zeroes=fill_zeroes)

    if 'scaler_mean' in data['val']:
        print('Dataset already has precomputed scalar, overwriting...')

    # Save as new dataset
    print('Compressing...')
    val_file = os.path.join(dataset_dir, 'val.npz')
    # Backup (but don't overwrite the backup)
    if not os.path.exists(val_file + '.bak'):
        copyfile(val_file, val_file + '.bak')
    np.savez_compressed(val_file, x=data['val']['x'], y=data['val']['y'],
                        x_offsets=data['val']['x_offsets'], y_offsets=data['val']['y_offsets'],
                        scaler_mean=scaler.mean, scaler_std=scaler.std, scaler_fillz=scaler.fill_zeroes)
    print('Added precomputed scaler to dataset')


class LazyDataLoader(dict):

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        # When to compute
        #  data['train_loader'] -> DataLoader(data['x_train'], data['y_train'], batch_size)
        if isinstance(value, tuple):
            category, batch_size, n_obs, scaler = value
            print('Lazyload', key)
            if 'loader' in key:
                # Lazy load the values needed to return the loader
                value = DataLoader(self['x_' + category], self['y_' + category], batch_size)
                # Cache computed value
                dict.__setitem__(self, key, value)
            elif key.endswith('_'+category):
                cat_data = np.load(os.path.join(self['dataset_dir'], category + '.npz'))
                data = {}
                data['x_' + category] = cat_data['x']
                data['y_' + category] = cat_data['y']
                if n_obs is not None:
                    data['x_' + category] = data['x_' + category][:n_obs]
                    data['y_' + category] = data['y_' + category][:n_obs]
                # Scale only the x vals
                data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
                # Store both the values of x and y
                value = data[key]
                dict.__setitem__(self, 'x_' + category, data['x_' + category])
                dict.__setitem__(self, 'y_' + category, data['y_' + category])
        return value

def lazy_load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, n_obs=None, fill_zeroes=True):
    print('loading dataset', dataset_dir)
    # These conditions have not been tested properly
    assert n_obs is None and fill_zeroes is False
    data = LazyDataLoader({'dataset_dir': dataset_dir})
    # Load the scaler from the validation set
    cat_data = np.load(os.path.join(dataset_dir, 'val.npz'))
    # If no precomputed scaler is embedded in the dataset, create it.
    if 'scalar_mean' not in cat_data.files:
        alter_dataset_precompute_scaler(dataset_dir,n_obs, fill_zeroes)
        cat_data = np.load(os.path.join(dataset_dir, 'val.npz'))

    scaler = StandardScaler(mean=float(cat_data['scaler_mean']), std=float(cat_data['scaler_std']), fill_zeroes=cat_data['scaler_fillz'])
    for category in ['train', 'val', 'test']:
        # polulate entries for preloading
        data[category + '_loader'] = (category, batch_size, n_obs, scaler)
        data['x_' + category] = (category, batch_size, n_obs, scaler)
        data['y_' + category] = (category, batch_size, n_obs, scaler)
    data['scaler'] = scaler
    print ('data', data)
    return data

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, n_obs=None, fill_zeroes=True):
    print('loading dataset', dataset_dir)
    data = LazyDataLoader({'dataset_dir': dataset_dir})
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std(), fill_zeroes=fill_zeroes)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

# Create a mask that is 1 for all values and 0 for all unknowns
def mask_nan_labels(labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    return mask

def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds-labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse

def evaluate_multiple_horizon(model, device, data, seq_length):
    scaler = data['scaler']
    realy = torch.Tensor(data['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
    # internally calls model.eval()
    test_met_df, yhat = calc_tstep_metrics(model, device, data['test_loader'], scaler, realy, seq_length)
    metric_vals = test_met_df.round(6)

    for horizon_i in [2, 5, 8, 11]:
        mae = metric_vals['mae'].iloc[horizon_i]
        mape = metric_vals['mape'].iloc[horizon_i]
        rmse = metric_vals['rmse'].iloc[horizon_i]
        print("Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                horizon_i + 1, mae, mape, rmse
            )
        )

def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def visualise_metrics(model, device, test_loader, scaler, realy, output_dir):
    # Compute predictions 1-12 horizons
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    # Compute error for every sensor for every timestep
    pred = scaler.inverse_transform(yhat).to(device)

    # print(pred.shape, realy.shape)
    # dcrnn_preds = np.load('/home/tijs/UvA/Thesis/Graph-Wavenet2/data/dcrnn_predictions.npz')['predictions']
    # pred = torch.Tensor(dcrnn_preds).to(device)
    # pred = pred.transpose(0, 1).transpose(1, 2)

    mask = mask_nan_labels(realy).to(device)
    # mae: [val_seq, sensors, timestep]
    mae = torch.abs(mask*pred - mask*realy).to(device)
    # mae per sensor per timestep
    mae_global = mae.mean(dim=0)
    # Compare variance between all sensors for specific timestep(s)   mae=[sensor, timestep]
    sensor_err_var = [torch.var(mae_global[:, t]) for t in [2, 5, 8, 11]]
    sensor_err_mean = [torch.mean(mae_global[:, t]) for t in [2, 5, 8, 11]]
    print (sensor_err_mean)
    print (sensor_err_var)

    # mae per sensor, now average over all timesteps
    global_error_per_sensor = np.array(mae_global.mean(dim=1).detach().cpu())
    np.savetxt(os.path.join(output_dir, 'sensors.txt'), global_error_per_sensor)

    fname = os.path.join(output_dir, 'sensor_preds.npz')
    # average mae over [val_seq, sensors]
    sensor_err = np.array(mae.mean(dim=2).detach().cpu())
    np.savez_compressed(fname, global_error_per_sensor=global_error_per_sensor, sensor_err=sensor_err)
    # for i in range(288):
    #     fname = os.path.join(output_dir, 'sensor_pred', 'sensors'+str(i)+'.txt')
    #     error_per_sensor = np.array(mae[i].mean(dim=1).detach().cpu())
    #     np.savetxt(fname, error_per_sensor)



def calc_tstep_metrics(model, device, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    # Compute predictions 1-12 horizons
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    # dcrnn_preds = np.load('/home/tijs/UvA/Thesis/Graph-Wavenet2/data/dcrnn_predictions.npz')['predictions']
    # pred2 = torch.Tensor(dcrnn_preds).to(device)
    # pred2 = pred2.transpose(0, 1).transpose(1, 2)

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        # pred = pred2[:, :, i]

        # This is kind of a trick, if the speeds are higher than 70 this needs to be changed.
        # A nicer way is to finetune it by the maximum/minimum value in the train set
        # pred = torch.clamp(pred, min=0., max=70.)
        #[idx, sensors, timeframe]
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real, null_val=0.0)])
    # For every horizon, metrics are a row in test_met
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def make_graph_inputs(args, device):
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports


def get_shared_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=40, help='Number of channels for internal conv')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--apt_size', default=10, type=int)
    parser.add_argument('--cat_feat_gc', action='store_true')
    parser.add_argument('--fill_zeroes', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='')
    return parser
