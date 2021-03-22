from main import main, cfg
import numpy as np
import time
import multiprocessing as mp


if __name__ == '__main__':

    attentions = [cfg.ModelType.baseline] * cfg.epochs + [cfg.ModelType.rnn] * cfg.epochs
    chunksize = (len(attentions) // cfg.cores) + 1

    start_time = time.time()
    with mp.Pool(processes=cfg.cores) as pool:
        results = pool.map(main, attentions, chunksize)
        base_counts = np.sum(results[:cfg.epochs])
        att_counts = np.sum(results[cfg.epochs:])

        print(f'Time: {time.time() - start_time} seconds where {(time.time() - start_time) // 60} minutes')
        print(f'baseline: {base_counts}/{cfg.epochs * cfg.test_episodes}')
        print(f'attention: {att_counts}/{cfg.epochs * cfg.test_episodes}')
