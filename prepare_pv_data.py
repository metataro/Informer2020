import argparse
import datetime
import logging
import os

import pandas as pd

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../../data/raw/export_solar_manager_data_133.csv', help='')
parser.add_argument('--out_path', default='../../data/informer/datasets', help='')
parser.add_argument('--train_size', default=0.8, type=float, help='')
parser.add_argument('--dev_size', default=0.1, type=float, help='')
parser.add_argument('--test_size', default=0.1, type=float, help='')
parser.add_argument('--gateway_ids', default=["5bc8d9a84ad9805f7c30c68d"], type=str, nargs='+', help='If not provided, all are used')
parser.add_argument('--random_seed', default=42, type=int, help='')

args = parser.parse_args()


def split_by_gateway(df: pd.DataFrame, train_size: float = 0.8, dev_size: float = 0.1, test_size: float = 0.1):
    assert train_size + dev_size + test_size == 1
    assert 'gateway_id' in df.columns

    # split into train, val, test
    # based on https://github.com/mozilla/CorporaCreator/blob/1d9be5e9279f4a2dcbf68e125fe9d614b69278e2/src/corporacreator/corpus.py
    train_size, dev_size, test_size = len(df) * train_size, len(df) * dev_size, len(df) * test_size

    gateway_index, uniques = pd.factorize(df['gateway_id'])
    df['gateway_index'] = gateway_index
    df_train = pd.DataFrame(columns=df.columns)
    df_dev = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)

    for i in range(max(gateway_index), -1, -1):
        df_i = df[df['gateway_index'] == i]
        if len(df_test) + len(df_i) <= test_size:
            df_test = pd.concat([df_test, df_i])
        elif len(df_dev) + len(df_i) <= dev_size:
            df_dev = pd.concat([df_dev, df_i])
        else:
            df_train = pd.concat([df_train, df_i])

    return df_train.drop('gateway_index', 1), df_dev.drop('gateway_index', 1), df_test.drop('gateway_index', 1)


def main():
    args = parser.parse_args()

    logging.info(f"Process {args.data_path}.")

    assert os.path.exists(args.data_path) and os.path.isfile(args.data_path)

    out_path = os.path.join(args.out_path, 'pv', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    assert not os.path.exists(out_path)

    os.makedirs(out_path)

    with open(os.path.join(out_path, 'args.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'--{k}={v}\n')

    df_raw = pd.read_csv(args.data_path)

    logging.info(f"Loaded {df_raw.shape[0]} samples with {df_raw.shape[1]} features")

    df_cleaned = df_raw.drop(df_raw[df_raw['pv_generation'] < 0].index)
    df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['сonsumption'] < 0].index)

    logging.info(f"Removed negative: {df_cleaned.shape[0]} of {df_raw.shape[0]} samples remain")

    df_cleaned = df_cleaned.groupby(['gateway_id', 'createdAt']).agg({'pv_generation': sum, 'сonsumption': sum})

    logging.info(f"Grouped by gateway_id and createdAt: {df_cleaned.shape[0]} of {df_raw.shape[0]} samples remain")

    df_cleaned = df_cleaned.reset_index()
    df_cleaned = df_cleaned.rename(columns={"createdAt": "date"})

    if len(args.gateway_ids) > 0:
        df_cleaned = df_cleaned[df_cleaned['gateway_id'].isin(args.gateway_ids)]

        logging.info(f"Filter gateway_id: {df_cleaned.shape[0]} of {df_raw.shape[0]} samples remain")

        df_cleaned.to_csv(os.path.join(out_path, 'all.csv'), index=False)
    else:
        df_train, df_valid, df_test = split_by_gateway(df_cleaned, args.train_size, args.dev_size, args.test_size)

        logging.info(f"Train/valid/test split: {df_train.shape}/{df_valid.shape}/{df_test.shape}")

        df_cleaned.to_csv(os.path.join(out_path, 'all.csv'), index=False)
        df_train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
        df_valid.to_csv(os.path.join(out_path, 'valid.csv'), index=False)
        df_test.to_csv(os.path.join(out_path, 'test.csv'), index=False)

    logging.info("Done.")


if __name__ == '__main__':
    main()
