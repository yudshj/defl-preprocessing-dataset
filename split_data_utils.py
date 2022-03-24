from collections import Counter
from pathlib import Path

import numpy as np


def get_split_results(distribution, x, y, verbose=False):
    label_all_images = [x[y == idx] for idx in range(10)]
    if verbose:
        print('Label distribution:', [len(i) for i in label_all_images])
    split_results = [[], [], [], []]
    for label, splits in distribution.items():
        a = np.cumsum(splits, dtype=np.float64)
        a = (a / a[-1] * len(label_all_images[label]))[:-1]
        a = np.round(a).astype(np.int64)
        res = np.split(label_all_images[label], a)
        for i, r in enumerate(res):
            for img in r:
                split_results[i].append((img, label))
    if verbose:
        for i, res in enumerate(split_results):
            counter = Counter(i for _, i in res)
            print(f'  node-{i} got {len(res)} samples', sorted(counter.items()))
    return split_results


def output_to_files(split_results, task: str, splita: str, suffix: str):
    output_dir = Path.cwd() / 'data' / 'splits' / task / splita
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, split_result in enumerate(split_results):
        node_x, node_y = zip(*split_result)
        node_x = np.array(node_x, dtype=np.uint8)
        node_y = np.array(node_y, dtype=np.uint8)
        np.savez_compressed(output_dir / f'node-{i}_x_{suffix}.npz', node_x)
        np.savez_compressed(output_dir / f'node-{i}_y_{suffix}.npz', node_y)