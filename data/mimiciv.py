from os import path
from typing import List, Optional

from pyhealth.datasets import MIMIC4Dataset

from pyhealth.tasks import mortality_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn

MIMIC_PATH = "./data/MIMIC4/hosp"


def get_mimic_iv(
        tables: Optional[List[str]] = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping: Optional[List[str]] = {"ICD9CM": "CCSCM"},
) -> MIMIC4Dataset:
    if not path.isdir(MIMIC_PATH):
        from personal_info import PhysioNet
        url = f"https://{PhysioNet.user_name}:{PhysioNet.password}@physionet.org/files/mimiciv/2.2/"
        # wget.download(f"{url}" , out=MIMIC_PATH)

    mimic4base = MIMIC4Dataset(
        root=MIMIC_PATH,
        tables=tables,
        code_mapping=code_mapping,
    )
    return mimic4base


def get_mortality_dataset():
    return get_mimic_iv().set_task(mortality_prediction_mimic3_fn)

def get_length_of_stay_dataset():
    return get_mimic_iv().set_task(length_of_stay_prediction_mimic3_fn)

if __name__ == "__main__":
    dataset = get_mimic_iv()
    print(dataset)