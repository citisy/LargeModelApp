from typing import Optional

from tqdm import tqdm

from .skeletons import Module


class StdOutCallback:
    pass


class TqdmVisCallback(Module):
    """
    Usage:
        Sequential(
            ...,

            iter_success_callbacks=[
                TqdmVisCallback()
            ]
        )
    """

    pbar: Optional

    def init(self, obj, **kwargs):
        # avoid to print when initialization
        self.pbar = tqdm(delay=1e-9)

    def on_process(self, obj: dict, **kwargs):
        self.pbar.update()
        return obj
