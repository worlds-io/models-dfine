"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data


class DetDataset(data.Dataset):
    # Cap on the per-worker decoded-image cache. Each worker process maintains its own
    # cache; with N persistent workers the total RSS cost is ~N × this × avg image size.
    # Set conservatively; see _cache_enabled() for when caching is actually on
    _MAX_CACHE_ITEMS = 5000

    # Caching only helps if the cache covers a useful fraction of the dataset — otherwise
    # almost every read is a miss, so the cache wastes RAM without improving hit rate.
    # Skip caching when the dataset is more than 50x the cache size (coverage < 2%)
    _CACHE_MIN_COVERAGE_RATIO = 50

    def _cache_enabled(self):
        return len(self) <= self._MAX_CACHE_ITEMS * self._CACHE_MIN_COVERAGE_RATIO

    def __getitem__(self, index):
        if not self._cache_enabled():
            img, target = self.load_item(index)
            if self.transforms is not None:
                img, target, _ = self.transforms(img, target, self)
            return img, target

        # Cache decoded images in memory after first load to avoid repeated disk I/O and
        # JPEG decoding. Effective for finetuning datasets that fit in RAM
        cache = getattr(self, '_item_cache', None)
        if cache is None:
            self._item_cache = {}
            cache = self._item_cache

        if index in cache:
            import copy
            img, target = copy.deepcopy(cache[index])
        else:
            img, target = self.load_item(index)
            if len(cache) < self._MAX_CACHE_ITEMS:
                import copy
                cache[index] = copy.deepcopy((img, target))

        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1
