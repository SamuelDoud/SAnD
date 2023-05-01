
from typing import Any, Dict, List, Optional

import numpy as np
from pyhealth.tokenizer import Tokenizer

from data.map import Map


class Tokenizers:
    def __init__(
        self,
        keys: List[str],
        dataset,
        depth: Optional[int] = None,
        mapping: Optional[Dict[str, str]] = {"drugs": "NDC", "conditions": "ICD9CM", "procedures": "ICD9PROC"}
    ):
        self.keys = keys
        self.tokenizers = {}
        self.depth = depth
        self.map = Map(mapping=mapping)
        for key in keys:
            alls = list(
                {
                    s
                    for l in [
                        self.map.get_ancestors(
                            sample[key][0], key=key, depth=self.depth
                        ) if depth else sample[key][0]
                        for sample in dataset.samples
                    ] 
                    for s in l
                }
            )
            self.tokenizers[key] = Tokenizer(alls)

    def to_data(self, sample: List[Any]) -> np.array:
        item_tables = []
        for key in self.keys:
            tokenizer = self.tokenizers[key]
            items = sample[key][0]
            ancestor_items = self.map.get_ancestors(items, key=key, depth=self.depth)
            item_table = np.zeros(shape=(tokenizer.get_vocabulary_size()))
            item_indicies = tokenizer.convert_tokens_to_indices(ancestor_items)
            item_table[item_indicies] = True
            item_tables.append(item_table)
        return np.concatenate(item_tables)

    def vocabulary_size(self) -> int:
        return sum(v.get_vocabulary_size() for v in self.tokenizers.values())
