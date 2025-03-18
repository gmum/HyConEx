from dataclasses import field, dataclass, fields
from enum import Enum
from typing import Type

from counterfactuals.datasets import *


class LossType(Enum):
    CE = 0
    MULTI_DISC = 1
    DEFAULT = 2
    PRETRAIN = 3


class DatasetType(Enum):
    MOONS = MoonsDataset
    MOONS3 = Moons3Dataset
    LAW = LawDataset
    AUDIT = AuditDataset
    HELOC = HelocDataset
    DIGITS = DigitsDataset
    BLOBS = BlobsDataset
    WINE = WineDataset
    ADULT = AdultDataset
    GERMAN = GermanCreditDataset
    OPENML = OpenmlDataset


@dataclass
class HyConExConfig:
    seed: int = field(default=0, metadata={"help": "Random seed."})
    output_dir: str = field(
        default="results", metadata={"help": "Directory to save the results."}
    )
    output_csv_file: str = field(
        default="results3.csv", metadata={"help": "CSV result file path location."}
    )
    model_load_path: str = field(
        default=".", metadata={"help": "Model weights load path."}
    )
    dataset: DatasetType = field(
        default=DatasetType.MOONS, metadata={
            "help": "Counterfactual dataset type."}
    )
    dataset_name: str = field(
        default="", metadata={"help": "Alias name for dataset."}
    )
    openml_id: str = field(default="", metadata={"help": "Openml dataset id."})

    device: str = field(default="cuda:0", metadata={"help": "Device type."})

    # Model parameters
    nr_blocks: int = field(
        default=4, metadata={"help": "Number of levels in the hypernetwork."}
    )
    hidden_size: int = field(
        default=256, metadata={"help": "Number of hidden units in the hypernetwork."}
    )
    dropout_rate: float = field(
        default=0.25, metadata={"help": "Training dropout rate."}
    )
    scheduler_t_mult: int = field(
        default=2, metadata={"help": "Multiplier for the scheduler."}
    )
    nr_restarts: int = field(
        default=1, metadata={"help": "Number of learning rate restarts."}
    )

    # Training parameters
    nr_epochs: int = field(default=1500, metadata={
                           "help": "Number of train epochs."})
    batch_size: int = field(default=256, metadata={
                            "help": "Dataloader batch size."})
    learning_rate: float = field(
        default=5e-4, metadata={"help": "Learning rate value."}
    )
    cluster_lambda: float = field(
        default=0.8,
        metadata={
            "help": "Lambda for the adjustment loss term of the closest cluster in the desired class."
        },
    )
    cluster_start_epoch: int = field(
        default=100,
        metadata={
            "help": "The epoch at which the cluster adjustment loss begins to be applied."
        },
    )
    pretrain: bool = field(
        default=False, metadata={"help": "Whether to pretrain base model or not."}
    )
    pretraining_epochs: int = field(
        default=500, metadata={"help": "Number of base model pretraining epochs."}
    )
    weight_decay: float = field(default=0.01, metadata={
                                "help": "Model weight decay."})
    use_distance: bool = field(
        default=False,
        metadata={
            "help": "Whether to use distance factor during counterfactual creation."
        },
    )
    early_stopping: bool = field(
        default=True,
        metadata={
            "help": "Whether to use early stopping or not."
        },
    )

    # Training second phase parameters
    loss_type: LossType = field(
        default=LossType.CE,
        metadata={
            "help": "Type of loss function used in counterfactuals training."},
    )

    class_lambda: float = field(
        default=0.8,
        metadata={"help": "Lambda for counterfactual classification loss term."},
    )
    dist_lambda: float = field(
        default=0.1, metadata={"help": "Lambda for counterfactual distance loss term."}
    )
    flow_lambda: float = field(
        default=0.1, metadata={"help": "Lambda for counterfactual flow loss term."}
    )

    class_start_epoch: int = field(
        default=500,
        metadata={
            "help": "The epoch at which the counterfactual classification loss begins to be applied."
        },
    )
    dist_start_epoch: int = field(
        default=400,
        metadata={
            "help": "The epoch at which the counterfactual distance loss begins to be applied."
        },
    )
    flow_start_epoch: int = field(
        default=400,
        metadata={
            "help": "The epoch at which the counterfactual flow loss begins to be applied."
        },
    )

    class_warm_up_epochs: int = field(
        default=300,
        metadata={
            "help": "Number of epochs required for the counterfactual classification loss"
            "to reach its maximum lambda value."
        },
    )
    dist_warm_up_epochs: int = field(
        default=200,
        metadata={
            "help": "Number of epochs required for the counterfactual distance loss to reach its maximum lambda value."
        },
    )
    flow_warm_up_epochs: int = field(
        default=200,
        metadata={
            "help": "Number of epochs required for the counterfactual flow loss to reach its maximum lambda value."
        },
    )

    # Evaluation parameters
    full_evaluation: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate all counterfactual methods."},
    )
    only_pred_eval: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate quality of prediction only."},
    )
    multiple_steps: bool = field(
        default=False,
        metadata={"help": "Whether to perform multiple-step version of HyConEx explanation."},
    )
    last_step_full: bool = field(
        default=True,
        metadata={"help": "Whether to fully perform last step in multiple-step explanation."},
    )
    eps: float = field(
        default=0.05,
        metadata={"help": "Hypernetwork weights factor for multiple-step version explanation."},
    )


def print_help(cfg_cls: Type[HyConExConfig]):
    """
    Print help information from dataclass metadata.

    Parameters:
        cfg_cls (Type[HyConExConfig]): Class configuration dataclass with 'help' metadata
    """
    for f in fields(cfg_cls):
        help_text = f.metadata.get("help", "No description available.")
        print(f"{f.name}: {help_text}")


subset_columns = [
    "name",
    "coverage",
    "validity",
    "proximity_continuous_manhattan",
    "proximity_continuous_euclidean",
    "prob_plausibility",
    "log_density_cf",
    "lof_scores_cf",
    "isolation_forest_scores_cf",
    "proximity_categorical_hamming",
    "proximity_categorical_jaccard",
    "time",
]
