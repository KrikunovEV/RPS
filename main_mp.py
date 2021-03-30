from main import run, cfg
import time
import multiprocessing as mp
import os
import pickle


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


def process_work(p_name: str, test_episodes: int, task_q: mp.Queue, result_q: mp.Queue):
    tasks_done = 0
    while True:
        task = task_q.get()

        if task.done:
            print(f'{p_name} done {tasks_done} tasks.')
            result_q.put(Result(coops=None, done=True, task=task))
            break

        coops = run(task.epoch, task.model_type)
        result_q.put(Result(coops=coops, done=False, task=task))
        tasks_done += 1
        print(f'{p_name}: model {task.model_type.name}, epoch {task.epoch}, coops {coops}/{test_episodes}')


if __name__ == '__main__':
    cfg.print_config()

    manager = mp.Manager()
    task_q = manager.Queue()
    result_q = manager.Queue()

    processes = []
    for process in range(cfg.cores):
        process_name = f'[C{process}]'
        args = (process_name, cfg.test_episodes, task_q, result_q)
        processes.append(mp.Process(target=process_work, args=args))
        processes[-1].start()

    start_time = time.time()

    for epoch in range(cfg.epochs):
        for model_type in cfg.mp_models:
            task_q.put(Task(epoch=epoch, model_type=model_type, done=False))

    for core in range(cfg.cores):
        task_q.put(Task(epoch=None, model_type=None, done=True))

    coops = dict(zip([model_type.name for model_type in cfg.mp_models], [0 for _ in cfg.mp_models]))
    epoch_counter = dict(zip([model_type.name for model_type in cfg.mp_models], [0 for _ in cfg.mp_models]))
    processes_done = 0
    total_epochs = 0
    while processes_done != cfg.cores:

        result = result_q.get()
        if result.done:
            processes_done += 1
            continue

        coops[result.task.model_type.name] += result.coops
        epoch_counter[result.task.model_type.name] += 1
        total_epochs += 1
        if total_epochs % 10 == 0:
            for model_type in cfg.mp_models:
                name = model_type.name
                print(f'{name}: {coops[name]}/{epoch_counter[name] * cfg.test_episodes} ('
                      f'{coops[name] / (epoch_counter[name] * cfg.test_episodes)})')
            if not os.path.exists(cfg.metric_directory):
                os.mkdir(cfg.metric_directory)
            filename = str(total_epochs) + '_' + cfg.pickle_file
            with open(os.path.join(cfg.metric_directory, filename), 'wb') as f:
                pickle.dump({'coops_dict': coops, 'epoch_counter_dict': epoch_counter}, f)

    print(f'Time: {time.time() - start_time} seconds where {(time.time() - start_time) // 60} minutes')
    for model_type in cfg.mp_models:
        name = model_type.name
        print(f'\nModel name: {name}')
        print(f'Epochs: {epoch_counter[name]} (should be {cfg.epochs})')
        print(f'Coops: {coops[name]}/{epoch_counter[name] * cfg.test_episodes}')
        print(f'Coops ratio: {coops[name] / (epoch_counter[name] * cfg.test_episodes)}')
