from typing import Dict, List, Optional

from pyhealth.medcode import InnerMap


class Map:
    def __init__(self, mapping: Optional[Dict[str, str]] = {"drugs": "NDC", "conditions": "ICD9CM", "procedures": "ICD9PROC"}) -> None:
        self.mapping = mapping
        self.innermaps = {key.lower(): InnerMap.load(code) for key, code in self.mapping.items()}

    def get_ancestors(self, datas: List[str], key: str, depth: Optional[int] = 0) -> List[str]:
        if not depth:
            return datas
        lower_key = key.lower()
        new_data = datas
        for data in datas:
            try:
                t = self.innermaps[lower_key].get_ancestors(data)
                safe_depth = min(depth, len(data) - 1)
                if t[safe_depth]:
                    new_data.extend(t[:depth])
            except Exception:
                continue
        return new_data
