
import torch

from model import (
    NetworkGNN,
    NODE_FEATURE_DIM,
    SECURITY_CLASSES,
)


def test_forward_single_graph():
    """Model must return three correctly shaped predictions."""

    model = NetworkGNN()
    model.eval()

    num_nodes = 5

    x = torch.randn(num_nodes, NODE_FEATURE_DIM)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.rand(6, 2)

    batch = torch.zeros(num_nodes, dtype=torch.long)

    with torch.no_grad():
        score, bottleneck, security = model(
            x,
            edge_index,
            edge_attr,
            batch,
        )

    assert score.shape == (1, 1)
    assert bottleneck.shape == (1, 2)
    assert security.shape == (1, SECURITY_CLASSES)

    assert torch.isfinite(score).all()
    assert torch.isfinite(bottleneck).all()
    assert torch.isfinite(security).all()


def test_forward_multiple_graphs():
    """Batch pooling must preserve separate graph predictions."""

    model = NetworkGNN()
    model.eval()

    # Graph 0: nodes 0, 1
    # Graph 1: nodes 2, 3, 4
    x = torch.randn(5, NODE_FEATURE_DIM)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.rand(4, 2)

    batch = torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    with torch.no_grad():
        score, bottleneck, security = model(
            x,
            edge_index,
            edge_attr,
            batch,
        )

    assert score.shape == (2, 1)
    assert bottleneck.shape == (2, 2)
    assert security.shape == (2, SECURITY_CLASSES)


def test_forward_graph_without_edges():
    """GNN must support isolated nodes."""

    model = NetworkGNN()
    model.eval()

    x = torch.randn(1, NODE_FEATURE_DIM)

    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    edge_attr = torch.empty(
        (0, 2),
        dtype=torch.float32,
    )

    batch = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        score, bottleneck, security = model(
            x,
            edge_index,
            edge_attr,
            batch,
        )

    assert score.shape == (1, 1)
    assert bottleneck.shape == (1, 2)
    assert security.shape == (1, SECURITY_CLASSES)

    assert torch.isfinite(score).all()


def test_checkpoint_save_and_load(tmp_path):
    """A saved checkpoint must reproduce model predictions."""

    model = NetworkGNN()
    model.eval()

    model_path = tmp_path / "test_model.pt"

    torch.save(model.state_dict(), model_path)

    loaded_model = NetworkGNN.from_pretrained(
        str(model_path)
    )

    loaded_model.eval()

    x = torch.randn(3, NODE_FEATURE_DIM)

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.rand(2, 2)

    batch = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        original = model(x, edge_index, edge_attr, batch)
        loaded = loaded_model(x, edge_index, edge_attr, batch)

    for original_output, loaded_output in zip(original, loaded):
        assert torch.allclose(
            original_output,
            loaded_output,
            atol=1e-6,
        )
