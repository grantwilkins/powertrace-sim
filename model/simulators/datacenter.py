from typing import List

from model.core.utils import load_classifier
from model.predictors.smooth_sampler import SmoothingSampler
from model.simulators.arrival_simulator import TokenSimulator


class Instance:
    def __init__(self, hw_type: str, tensor_parallelism: int, model: str):
        self.hw_type = hw_type
        self.tensor_parallelism = tensor_parallelism
        self.model = model
        self.classifier = load_classifier(
            path=f"./gru_classifier_weights/{model}_tp{tensor_parallelism}_{hw_type}.pt",
            device="cpu",
        )
        # TODO: Update this to not need the data file in full
        self.smoothing_sampler = SmoothingSampler(
            state_stats=self.classifier.state_stats
        )

        self.arrival_simulator = TokenSimulator(
            model=self.model,
            tensor_parallelism=self.tensor_parallelism,
            hardware_accelerator=self.hw_type,
        )


class Node:
    def __init__(
        self, hw_type: str, num_gpus: int = 8, instances: List[Instance] = None
    ):
        self.hw_type = hw_type
        self.num_gpus = num_gpus
        self.instances = instances

        total_tp = sum(instance.tensor_parallelism for instance in instances)
        if total_tp > num_gpus:
            raise ValueError(
                f"Total tensor parallelism {total_tp} exceeds number of GPUs {num_gpus} on node."
            )


class Rack:
    def __init__(self, hw_type: str, num_nodes: int = 4):
        self.hw_type = hw_type
        self.num_servers = num_nodes
        self.servers = [self._create_server(i) for i in range(num_nodes)]

    def _create_server(self, server_id: int):
        return {"id": server_id, "type": self.hw_type, "status": "active"}
