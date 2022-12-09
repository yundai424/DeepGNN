# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""AML."""
from typing import List, Any, Dict, Optional
from dataclasses import dataclass, field
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.graph_engine.snark.local import Client
from deepgnn.graph_engine.snark.distributed import Client as DistributedClient

from azureml.core import Workspace
from ray_on_aml.core import Ray_On_AML

from deepgnn.graph_engine import Graph
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel, BaseUnsupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder, SageEncoder

from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    GENodeSampler,
    SamplingStrategy,
)
import deepgnn.graph_engine.snark.server as server
import deepgnn.graph_engine.snark.client as client
from azureml.core import Workspace
from ray_on_aml.core import Ray_On_AML


@dataclass
class PTGSupervisedGraphSageQuery:
    """GAT Query."""

    feature_meta: list # = field(default_factory=lambda: np.array([[0, 1433]]))
    label_meta: list # = field(default_factory=lambda: np.array([[1, 1]]))
    feature_type: np.dtype = np.float32
    label_type: np.dtype = np.float32
    edge_type: int = 0
    fanouts: list = field(default_factory=lambda: np.array([[25, 25]]))

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Query graph for training data."""
        context = {"inputs": np.array(inputs)}

        #context["0"] = np.ones((len(inputs), 30000))

        context["label"] = graph.node_features(
            context["inputs"],
            self.label_meta,
            self.label_type,
        )

        n2_out = context["inputs"]  # Output nodes of 2nd (final) layer of convolution
        # input nodes of 2nd layer of convolution (besides the output nodes themselves)
        n2_in = graph.sample_neighbors(n2_out, self.edge_type, self.fanouts[1])[
            0
        ].flatten()
        #  output nodes of first layer of convolution (all nodes that affect output of 2nd layer)
        n1_out = np.concatenate([n2_out, n2_in])
        # input nodes to 1st layer of convolution (besides the output)
        n1_in = graph.sample_neighbors(n1_out, self.edge_type, self.fanouts[0])[
            0
        ].flatten()
        # Nodes for which we need features (layer 0)
        n0_out = np.concatenate([n1_out, n1_in])
        x0 = graph.node_features(
            n0_out, self.feature_meta, self.feature_type
        )

        context["x0"] = x0.reshape((context["inputs"].shape[0], -1, self.feature_meta[0, 1]))
        context["out_1"] =  np.array([n1_out.shape[0]] * context["inputs"].shape[0])  # Number of output nodes of layer 1
        context["out_2"] =  np.array([n2_out.shape[0]] * context["inputs"].shape[0])  # Number of output nodes of layer 2
        return context


class PTGSupervisedGraphSage(BaseSupervisedModel):
    """Supervised graphsage model implementation with torch geometric."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_type: np.dtype,
        feature_idx: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
        metric: BaseMetric = MRR(),
        feature_enc: Optional[FeatureEncoder] = None,
    ):
        """Initialize a graphsage model for node classification."""
        super(PTGSupervisedGraphSage, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        # only 2 hops are allowed.
        assert len(fanouts) == 2
        self.fanouts = fanouts
        self.edge_type = edge_type

        conv_model = pyg_nn.SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(feature_dim, embed_dim))
        self.convs.append(conv_model(embed_dim, embed_dim))

        self.label_idx = label_idx
        self.label_dim = label_dim
        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        self.metric = metric
        nn.init.xavier_uniform_(self.weight)

    def build_edges_tensor(self, N, K):
        """Build edge matrix."""
        nk = torch.arange((N * K).item(), dtype=torch.long, device=N.device)
        src = (nk // K).reshape(1, -1)
        dst = (N + nk).reshape(1, -1)
        elist = torch.cat([src, dst], dim=0)
        return elist

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate scores for a list of nodes."""
        self.encode_feature(context)
        embeds = self.get_embedding(context)
        scores = torch.matmul(embeds, self.weight)
        return scores

    def metric_name(self):
        """Metric used for training."""
        return self.metric.name()

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate embedding."""
        out_1 = context["out_1"][0]  # TODO 0 valid for multiple blocks?
        out_2 = context["out_2"][0]
        try:
            out_1 = out_1[0]
            out_2 = out_2[0]
        except IndexError:
            pass
        edges_1 = self.build_edges_tensor(out_1, self.fanouts[0])  # Edges for 1st layer

        # TODO note reshape
        x1 = self.convs[0](context["x0"].reshape((-1, context["x0"].shape[-1])), edges_1)[
            :out_1, :
        ]  # Output of 1st layer (cut out 2 hop nodes)
        x1 = F.relu(x1)
        edges_2 = self.build_edges_tensor(out_2, self.fanouts[1])  # Edges for 2nd layer
        x2 = self.convs[1](x1, edges_2)[
            :out_2, :
        ]  # Output of second layer (nodes for which loss is computed)
        x2 = F.relu(x2)
        return x2

    def _loss_inner(self, context):
        """Cross entropy loss for a list of nodes."""
        if isinstance(context, dict):
            labels = context["label"].squeeze()  # type: ignore
        elif isinstance(context, torch.Tensor):
            labels = context.squeeze()  # type: ignore
        else:
            raise TypeError("Invalid input type.")
        device = labels.device

        # TODO(chaoyl): Due to the bug of pytorch argmax, we have to copy labels to numpy for argmax
        # then copy back to Tensor. The fix has been merged to pytorch master branch but not included
        # in latest stable version. Revisit this part after updating pytorch with the fix included.
        # issue: https://github.com/pytorch/pytorch/issues/32343
        # fix: https://github.com/pytorch/pytorch/pull/37864
        labels = labels.cpu().numpy().argmax(1)
        scores = self.get_score(context)
        return (
            # self.xent(
            #    scores,
            #    Variable(torch.tensor(labels.squeeze(), dtype=torch.int64).to(device)),
            # ),
            scores,
            scores.argmax(dim=1),
            torch.tensor(labels.squeeze(), dtype=torch.int64),
        )

    def forward(self, context):
        """Return cross entropy loss."""
        return self._loss_inner(context)


address = [f"localhost:9990", f"localhost:9991"]

import random
@ray.remote(scheduling_strategy="SPREAD")  # double check
class Counter:
    def __init__(self):
        self.g = DistributedClient([address[random.random() > .5]])  #Client("/tmp/reddit", [0])
        self.query_obj = PTGSupervisedGraphSageQuery(
            label_meta=np.array([[0, 50]]),
            feature_meta=np.array([[1, 300]]),
            feature_type=np.float32,
            edge_type=0,
            fanouts=[5, 5],
        )

    def call(self, batch):
        return self.query_obj.query(self.g, batch)


class Counter2:
    def __init__(self):
        self.g = DistributedClient([address[0]])  #Client("/tmp/reddit", [0])
        self.query_obj = PTGSupervisedGraphSageQuery(
            label_meta=np.array([[0, 50]]),
            feature_meta=np.array([[1, 300]]),
            feature_type=np.float32,
            edge_type=0,
            fanouts=[5, 5],
        )

    def __call__(self, batch):
        return self.query_obj.query(self.g, batch)


def train_func(config: Dict):
    """Train function main."""
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    s = [
        server.Server("/tmp/reddit", [0, 1], address[0]),
        #server.Server("/tmp/reddit", [0, 1], address[1])
    ]

    model = PTGSupervisedGraphSage(
        num_classes=50,
        label_idx=0,
        label_dim=50,
        feature_type=np.float32,
        feature_idx=1,
        feature_dim=300,
        edge_type=0,
        fanouts=[5, 5],
    )
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    optimizer = train.torch.prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()

    SAMPLE_NUM = 152410
    BATCH_SIZE = 512

    dataset = ray.data.range(SAMPLE_NUM - (SAMPLE_NUM % BATCH_SIZE), parallelism=-1).repartition(SAMPLE_NUM // BATCH_SIZE)
    #dataset = ray.data.read_text("/tmp/reddit/notes.train").repartition(SAMPLE_NUM // BATCH_SIZE)
    print(dataset)
    pipe = dataset.window(blocks_per_window=4).repeat(5)  # map batches gets run once per each window full reset for actor pool

    def transform(batch):
        g = DistributedClient([f"localhost:9990"])
        query_obj = PTGSupervisedGraphSageQuery(
            label_meta=np.array([[0, 50]]),
            feature_meta=np.array([[1, 300]]),
            feature_type=np.float32,
            edge_type=0,
            fanouts=[5, 5],
        )
        output = query_obj.query(g, batch)
        return output

    pipe = pipe.map_batches(transform)
    """
    worker = Counter.remote()#num_cpus=.5)
    pipe = pipe.map_batches(lambda batch: ray.get(worker.call.remote(batch)), batch_size=BATCH_SIZE)
    """
    """
    worker1, worker2 = Counter.remote(), Counter.remote() 
    pool = ray.util.actor_pool.ActorPool([worker1, worker2]) 
    #
    def transform(batch):
        pool.submit(lambda a, b: a.call.remote(b), batch)
        return pool.get_next()

    pipe = pipe.map_batches(transform, batch_size=BATCH_SIZE)
    """
    """
    # Not work, foreach cant iterate over dataset must transform it inplace
    worker1, worker2 = Counter.remote(), Counter.remote() 
    pool = ray.util.actor_pool.ActorPool([worker1, worker2]) 
    #
    def transform(dataset):
        #pool.submit(lambda a, b: a.call.remote(b), batch)
        assert False, dataset
        assert False, list(pool.map(lambda a, b: a.call.remote(b), dataset.iter_batches()))
        return ray.data.from_items(list(pool.map(lambda a, b: a.call.remote(b), dataset.iter_batches())))

    pipe = pipe.foreach_window(transform)
    """
    """
    pipe = pipe.map_batches(Counter2(), compute=ray.data.ActorPoolStrategy(2, 2))
    """
    """
    g = DistributedClient([address])
    query_obj = PTGSupervisedGraphSageQuery(
        label_meta=np.array([[0, 50]]),
        feature_meta=np.array([[1, 300]]),
        feature_type=np.float32,
        edge_type=0,
        fanouts=[5, 5],
    )

    def transform(batch):
        return query_obj.query(g, batch)
    pipe = pipe.map_batches(transform)
    """
    """
    g = Client("/tmp/reddit", [0])
    query_obj = PTGSupervisedGraphSageQuery(
        label_meta=np.array([[0, 50]]),
        feature_meta=np.array([[1, 300]]),
        feature_type=np.float32,
        edge_type=0,
        fanouts=[5, 5],
    )
    N_WORKERS = 2
    dataset = TorchDeepGNNDataset(
        sampler_class=GENodeSampler,
        backend=type("Backend", (object,), {"graph": g})(),  # type: ignore
        sample_num=SAMPLE_NUM,
        num_workers=1,  # sampler sample count is divided by num_workers and data parralele count
        worker_index=0,
        node_types=np.array([0], dtype=np.int32),
        batch_size=BATCH_SIZE,
        query_fn=query_obj.query,
        strategy=SamplingStrategy.RandomWithoutReplacement,
        prefetch_queue_size=10,
        prefetch_worker_size=N_WORKERS,
    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=N_WORKERS,
    )
    for epoch in range(5):
        metrics = []
        losses = []
        for i, batch in enumerate(dataset):
            """
    for epoch, epoch_pipe in enumerate(pipe.iter_epochs()):
        metrics = []
        losses = []
        for i, batch in enumerate(
                epoch_pipe.iter_torch_batches(prefetch_blocks=0, batch_size=BATCH_SIZE)
            ):
            scores = model(batch)[0]
            labels = batch["label"].squeeze().argmax(1).detach()

            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.append((scores.squeeze().argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            #if i >= SAMPLE_NUM / BATCH_SIZE / session.get_world_size():
            #    break

        try:
            pass#print(epoch_pipe.stats())
        except NameError:
            pass

        print("RESULTS:!", np.mean(metrics), np.mean(losses), "N_BATCHES", i)

        session.report(
            {
                "metric": np.mean(metrics),
                "loss": np.mean(losses),
            }
        )

ws = Workspace.from_config("../config.json")
ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="gpu-cluster", maxnode=2)
ray = ray_on_aml.getRay(gpu_support=True)

# ray.init()
"""
# Only works for dedicated gpus, while this does not have it
trainer = TorchTrainer(
    train_func,
    train_loop_config={},
    run_config=RunConfig(),
    scaling_config=ScalingConfig(num_workers=1, resources_per_worker={"CPU": 1, "GPU": 1}, use_gpu=True),
)
result = trainer.fit()
"""
trainer = TorchTrainer(
    train_func,
    train_loop_config={},
    run_config=RunConfig(),
    scaling_config=ScalingConfig(
        num_workers=1, resources_per_worker={"CPU": 1, "GPU": 0}, use_gpu=False
    ),
)
result = trainer.fit()

# ray_on_aml.shutdown()