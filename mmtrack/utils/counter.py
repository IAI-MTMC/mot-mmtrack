# Copyright (c) OpenMMLab. All rights reserved.
import time

import torch


class counter:

    def __init__(self, task_name: str):
        self.task_name = task_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, *args, **kwargs):
        torch.cuda.synchronize()
        self.end = time.perf_counter()
        print(
            f'[{self.task_name}] takes {self.end - self.start:.3f} seconds\n')
