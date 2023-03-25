from os import path
from typing import List, Optional
import wget

from pyhealth.datasets import MIMIC3Dataset

from personal_info import PhysioNet

MIMIC_PATH = "./MIMIC3"

def get_mimic_iii(
        tables: Optional[List[str]] = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping: Optional[List[str]] = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}
    ) -> MIMIC3Dataset:
    if not path.isdir("./MIMIC3"):
        url = f"https://{PhysioNet.user_name}:{PhysioNet.password}@physionet.org/files/mimiciii/1.4/"
        wget.download(f"{url} -o {MIMIC_PATH}")

    mimic3base = MIMIC3Dataset(
        root=MIMIC_PATH,
        tables=tables,
        code_mapping=code_mapping,
    )
    return mimic3base

dataset = get_mimic_iii()

print(dataset.stat())
print(dataset.info())