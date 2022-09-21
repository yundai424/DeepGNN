# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Any, Iterator
import os
import sys
import pytest
import tempfile
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from deepgnn.graph_engine.snark.local import Client
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import set_seed
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from deepgnn.graph_engine.data.citation import Cora
from model import GAT, GATQueryParameter  # type: ignore
from deepgnn.graph_engine import Graph, graph_ops

class DeepGNNDataset(Dataset):
    """Cora dataset with file sampler."""
    def __init__(self, data_dir: str, node_types: List[int], feature_meta: List[int], label_meta: List[int], feature_type: np.dtype, label_type: np.dtype, neighbor_edge_types: List[int] = [0], num_hops: int = 2):
        self.g = Client(data_dir, [0, 1])
        self.node_types = np.array(node_types)
        self.feature_meta = np.array([feature_meta])
        self.label_meta = np.array([label_meta])
        self.feature_type = feature_type
        self.label_type = label_type
        self.neighbor_edge_types = np.array(neighbor_edge_types, np.int64)
        self.num_hops = num_hops
        self.count = self.g.node_count(self.node_types)

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Query used to generate data for training."""
        inputs = np.array(idx, np.int64)
        nodes, edges, src_idx = graph_ops.sub_graph(
            self.g,
            inputs,
            edge_types=self.neighbor_edge_types,
            num_hops=self.num_hops,
            self_loop=True,
            undirected=True,
            return_edges=True,
        )
        input_mask = np.zeros(nodes.size, np.bool)
        input_mask[src_idx] = True

        feat = self.g.node_features(nodes, self.feature_meta, self.feature_type)
        label = self.g.node_features(nodes, self.label_meta, self.label_type)
        label = label.astype(np.int32)
        edges_value = np.ones(edges.shape[0], np.float32)
        edges = np.transpose(edges)
        adj_shape = np.array([nodes.size, nodes.size], np.int64)

        return nodes, feat, input_mask, label, edges, edges_value, adj_shape


class FileSampler(Sampler[int]):
    def __init__(self, filename: str):
        self.filename = filename

    def __len__(self) -> int:
        raise NotImplementedError("")

    def __iter__(self) -> Iterator[int]:
        with open(self.filename, "r") as file:
            x = []
            for i, line in enumerate(file.readlines()):
                x.append(int(line))
                if i % 512 == 5122:
                    yield np.array(x)
                    x = []


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def test_pytorch_gat_cora():
    set_seed(123)
    g = Cora()
    qparam = GATQueryParameter(
        neighbor_edge_types=np.array([0], np.int32),
        feature_idx=0,
        feature_dim=g.FEATURE_DIM,
        label_idx=1,
        label_dim=1,
    )
    model = GAT(
        in_dim=g.FEATURE_DIM,
        head_num=[8, 1],
        hidden_dim=8,
        num_classes=g.NUM_CLASSES,
        ffd_drop=0.6,
        attn_drop=0.0,
        q_param=qparam,
    )

    args = argparse.Namespace(
        data_dir=g.data_dir(),
        converter=DataConverterType.LOCAL,
    )
    dataset = DeepGNNDataset(g.data_dir(), [0, 1, 2], [0, g.FEATURE_DIM], [1, 1], np.float32, np.float32)

    def create_dataset():
        return DataLoader(dataset, sampler=FileSampler(os.path.join(g.data_dir(), "train.nodes")), batch_size=1)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005,
        weight_decay=0.0005,
    )
    num_epochs = 200
    ds = create_dataset()
    model.train()
    for ei in range(num_epochs):
        for si, batch_input in enumerate(ds):
            optimizer.zero_grad()
            loss, pred, label = model(batch_input)
            loss.backward()
            optimizer.step()
            acc = model.compute_metric([pred], [label])
            get_logger().info(
                f"epoch {ei} - {si}, loss {loss.data.item() :.6f}, accuracy {acc.data.item():.6f}"
            )

    def create_eval_dataset():
        return DataLoader(dataset, sampler=FileSampler(os.path.join(g.data_dir(), "test.nodes")), batch_size=1)

    test_dataset = create_eval_dataset()
    model.eval()
    for si, batch_input in enumerate(test_dataset):
        loss, pred, label = model(batch_input)
        acc = model.compute_metric([pred], [label])
        get_logger().info(
            f"evaluate loss {loss.data.item(): .6f}, accuracy {acc.data.item(): .6f}"
        )
        assert acc.data.item() >= 0.825


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
