from typing import Dict, List, Optional

from pyhealth.medcode import InnerMap


class Map:
    def __init__(self, mapping: Optional[Dict[str, str]] = {"drugs": "NDC", "conditions": "ICD9CM", "procedures": "ICD9PROC"}) -> None:
        self.mapping = mapping
        self.innermaps = {key.lower(): InnerMap.load(code) for key, code in self.mapping.items()}

    def get_ancestors(self, datas: List[str], key: str, depth: Optional[int] = 0) -> List[str]:
        if not depth:
            return datas

        new_data = []
        for data in datas:
            try:
                t = self.innermaps[key.lower()].get_ancestors(data)
                if t[depth - 1]:
                    new_data.append(t[depth - 1])
                else:
                    new_data.append(data)
            except Exception:
                new_data.append(data)
        return new_data
