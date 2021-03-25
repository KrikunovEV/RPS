from main import main, cfg
import time
import multiprocessing as mp


class Task:
    def __init__(self, epoch: int, model_type: cfg.ModelType, done: bool):
        self.epoch = epoch
        self.model_type = model_type
        self.done = done


class Result:
    def __init__(self, coops: int, done: bool, task: Task):
        self.coops = coops
        self.done = done
        self.task = task


def process_work(name: str, test_episodes: int, log: cfg.LogType, task_q: mp.Queue, result_q: mp.Queue):
    tasks_done = 0
    while True:
        task = task_q.get()

        if task.done:
            print(f'{name} done {tasks_done} tasks.')
            result_q.put(Result(0, True, task))
            break

        coops = main(task.epoch, task.model_type, log)
        result_q.put(Result(coops, False, task))
        tasks_done += 1
        print(f'{name}: model {task.model_type.name}, epoch {task.epoch}, coops {coops}/{test_episodes}')


if __name__ == '__main__':
    if not cfg.Train:
        print(f'The value of cfg.Train is {cfg.Train}. Is this expected?')

    manager = mp.Manager()
    task_q = manager.Queue()
    result_q = manager.Queue()

    processes = []
    for process in range(cfg.cores):
        process_name = f'[C{process}]'
        args = (process_name, cfg.test_episodes, cfg.LogType.local, task_q, result_q)
        processes.append(mp.Process(target=process_work, args=args))
        processes[-1].start()

    start_time = time.time()

    for epoch in range(cfg.epochs):
        for model_type in cfg.ModelType:
            task_q.put(Task(epoch, model_type, False))
    for core in range(cfg.cores):
        task_q.put(Task(None, None, True))

    coops = dict(zip([model_type.name for model_type in cfg.ModelType], [0 for _ in cfg.ModelType]))
    epoch_counter = dict(zip([model_type.name for model_type in cfg.ModelType], [0 for _ in cfg.ModelType]))
    processes_done = 0
    while processes_done != cfg.cores:

        result = result_q.get()
        if result.done:
            processes_done += 1
            continue

        coops[result.task.model_type.name] += result.coops
        epoch_counter[result.task.model_type.name] += 1
        if epoch_counter[result.task.model_type.name] % 10 == 0:
            for model_type in cfg.ModelType:
                name = model_type.name
                print(f'{name}: {coops[name]}/{epoch_counter[name] * cfg.test_episodes} ('
                      f'{coops[name] / (epoch_counter[name] * cfg.test_episodes)})')

    print(f'Time: {time.time() - start_time} seconds where {(time.time() - start_time) // 60} minutes')
    for model_type in cfg.ModelType:
        name = model_type.name
        print(f'{name}: {coops[name]}/{cfg.epochs * cfg.test_episodes} ('
              f'{coops[name] / (cfg.epochs * cfg.test_episodes)})')
