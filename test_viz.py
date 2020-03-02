import torch
import numpy as np
import pandas as pd
import time
import util
from engine import Trainer
import os
from durbango import pickle_save
from fastprogress import progress_bar

from model import GWNet
from model_gwnet import gwnet
from model_lstm import LSTMNet
from model_static import StaticNet
from util import *
from util import calc_tstep_metrics
from exp_results import summary


def main(args, **model_kwargs):
    # Train on subset of sensors (faster for isolated pred)
    # incl_sensors = list(range(207)) #[17, 111, 12, 80, 200]
    # args.num_sensors = len(incl_sensors)
    device = torch.device(args.device)
    # WARN Careful! Graph wavenet has been trained without fill zeroes in its scalar
    data = util.lazy_load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    scaler = data['scaler']
    supports = []
    aptinit = 0
    # aptinit, supports = util.make_graph_inputs(args, device)

    # Length of the prediction
    args.seq_length = data['y_val'].shape[1]
    args.num_sensors = data['x_val'].shape[2]
    if args.static:
        print('Selected static prediction')
        model = StaticNet.from_args(args, device, supports, aptinit, **model_kwargs)
    elif args.lstm:
        print('Selected LSTM-FC model')
        args.nhid = 256
        args.weight_decay = 0.0005
        args.learning_rate = 0.001
        model = LSTMNet.from_args(args, device, supports, aptinit, **model_kwargs)
    else:
        print('Selected Graph Wavenet model')
        # Params: ---graph_wavenet --data data/METR-LA --checkpoint pretrained/graph_wavenet_repr.pth --nhid 32 --do_graph_conv --addaptadj --device cuda:0 --in_dim=2 --save experiment
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        supports = [torch.tensor(i).to(device) for i in adj_mx]

        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None

        model = gwnet(device, num_nodes=args.num_nodes, dropout=args.dropout, supports=supports, gcn_bool=args.do_graph_conv,
                      addaptadj=args.addaptadj,
                      aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid,
                      dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)

    print(args)

    if args.checkpoint:
        model.load_checkpoint(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    print (scaler)

    # Only the speeds?
    realy = torch.Tensor(data['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
    print('visualising frames')
    visualise_metrics(model, device, data['test_loader'], scaler, realy, args.save)
    evaluate_multiple_horizon(model, device, data, args.seq_length)


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    # Added options to train with different architectures
    parser.add_argument('--graph_wavenet', action='store_true', help='Train the Graph Wavenet setting')
    parser.add_argument('--lstm', action='store_true', help='Train the LSTM-FC setting')
    parser.add_argument('--isolated_sensors', action='store_true', help='Train every sensor independently.')
    parser.add_argument('--static', action='store_true', help='Do static prediction')
    parser.add_argument('--save', type=str, default='experiment', help='save path')

    args = parser.parse_args()
    t1 = time.time()
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    pickle_save(args, f'{args.save}/args.pkl')
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} seconds")