import numpy as np
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
import torch

from data.mimiciii import get_mortality_dataset

dataset = get_mortality_dataset()
np.random.seed(1234)
batch_size = 128

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])

train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    dataset=dataset,
    feature_keys=["conditions", "procedures", "drugs"],
    label_key="label",
    embedding_dim=128,
    mode="binary",
)
model.to(device)

trainer = Trainer(
    model=model, metrics=["accuracy"], device=device, exp_name="mortality_prediction"
)

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=20,
    monitor="accuracy",
    monitor_criterion="max",
    load_best_model_at_last=True,
)

result = trainer.evaluate(test_loader)
print(result)

y_true, y_prob, loss = trainer.inference(test_loader)

result = multilabel_metrics_fn(
    y_true,
    y_prob,
    metrics=[
        "pr_auc_samples",
        "f1_weighted",
        "recall_macro",
        "precision_micro",
    ],
)

print(result)
