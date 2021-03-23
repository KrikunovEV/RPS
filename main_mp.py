from main import main, cfg
import numpy as np
import time
import multiprocessing as mp


if __name__ == '__main__':

    inputs = [cfg.ModelType.baseline] * cfg.epochs + [cfg.ModelType.rnn]\
                 * cfg.epochs + [cfg.ModelType.attention] * cfg.epochs
    chunksize = (len(inputs) // cfg.cores) + 1

    start_time = time.time()
    with mp.Pool(processes=cfg.cores) as pool:
        results = pool.map(main, inputs, chunksize)
        base_counts = np.sum(results[:cfg.epochs])
        rnn_counts = np.sum(results[cfg.epochs:cfg.epochs * 2])
        att_counts = np.sum(results[cfg.epochs * 2:])

        print(f'Time: {time.time() - start_time} seconds where {(time.time() - start_time) // 60} minutes')
        print(f'baseline: {base_counts}/{cfg.epochs * cfg.test_episodes}')
        print(f'rnn: {rnn_counts}/{cfg.epochs * cfg.test_episodes}')
        print(f'attention: {att_counts}/{cfg.epochs * cfg.test_episodes}')
