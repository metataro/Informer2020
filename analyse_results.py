import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import metric

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default='./results/informer_custom_ftS_sl720_ll168_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_pv_0/', help='')

args = parser.parse_args()


def main():
    args = parser.parse_args()

    logging.info(f"Process {args.results_path}.")

    assert os.path.exists(args.results_path) and os.path.isdir(args.results_path)

    metrics = np.load(os.path.join(args.results_path, 'metrics.npy'))
    preds = np.load(os.path.join(args.results_path, 'pred.npy'))
    trues = np.load(os.path.join(args.results_path, 'true.npy'))

    lags = np.arange(trues.shape[1]) + 1
    maes = np.zeros_like(lags, dtype=np.float32)
    mses = np.zeros_like(lags, dtype=np.float32)

    for i in range(trues.shape[1]):
        mae, mse, rmse, mape, mspe = metric(preds[:, i].flatten(), trues[:, i].flatten())
        maes[i] = mae
        mses[i] = mse

    plt.figure()
    plt.plot(lags, maes, marker='o', label="mae")
    plt.plot(lags, np.mean(preds, axis=0).flatten(), marker='o', label="preds")
    plt.plot(lags, np.mean(trues, axis=0).flatten(), marker='o', label="trues")
    plt.legend()
    plt.savefig(os.path.join(args.results_path, 'mae.png'), bbox_inches='tight')

    plt.figure()
    plt.plot(lags, mses, marker='o', label="mse")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(args.results_path, 'mse.png'), bbox_inches='tight')


if __name__ == '__main__':
    main()
