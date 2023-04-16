from os import path
from typing import List, Optional

from pyhealth.datasets import MIMIC3Dataset

from pyhealth.tasks import mortality_prediction_mimic3_fn

MIMIC_PATH = "./data/MIMIC3"


def get_mimic_iii(
        tables: Optional[List[str]] = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping: Optional[List[str]] = {"ICD9CM": "CCSCM"},
) -> MIMIC3Dataset:
    if not path.isdir(MIMIC_PATH):
        from personal_info import PhysioNet
        url = f"https://{PhysioNet.user_name}:{PhysioNet.password}@physionet.org/files/mimiciii/1.4/"
        # wget.download(f"{url}" , out=MIMIC_PATH)

    mimic3base = MIMIC3Dataset(
        root=MIMIC_PATH,
        tables=tables,
        code_mapping=code_mapping,
    )
    return mimic3base


def get_mortality_dataset():
    return get_mimic_iii().set_task(mortality_prediction_mimic3_fn)
