from data.mimiciii import get_mimic_iii


dataset = get_mimic_iii()

print(dataset.stat())
print(dataset.info())