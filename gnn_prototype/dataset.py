import json
import os
from typing import List, Dict, Any
import torch
import torch_geometric.data
from torch_geometric.data import Dataset

class NetworkDataset(Dataset):
    """
    PyTorch Geometric Dataset for network topology prediction.
    Loads graph data from JSON files in the 'data/' directory.
    """
    NODE_TYPES = ["router", "switch", "printer", "host", "firewall", "ids_ips", "l3_switch"]
    NUM_NODE_TYPES = len(NODE_TYPES)

    def __init__(self, root: str):
        super().__init__() # Use super().__init__() for Dataset base class
        self.data_files = self._load_json_files(root)
        self.processed_data = []
        self._process_all_data()

    @staticmethod
    def _normalize_attributes(bandwidths: List[float], throughputs: List[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalizes bandwidth and throughput using min-max scaling."""
        # Bandwidth normalization
        bw_tensor = torch.tensor(bandwidths, dtype=torch.float32).unsqueeze(1)
        min_bw, max_bw = bw_tensor.min(), bw_tensor.max()
        if max_bw - min_bw > 0:
            normalized_bw = (bw_tensor - min_bw) / (max_bw - min_bw)
        else:
            normalized_bw = torch.zeros_like(bw_tensor)

        # Throughput normalization
        tp_tensor = torch.tensor(throughputs, dtype=torch.float32).unsqueeze(1)
        min_tp, max_tp = tp_tensor.min(), tp_tensor.max()
        if max_tp - min_tp > 0:
            normalized_tp = (tp_tensor - min_tp) / (max_tp - min_tp)
        else:
            normalized_tp = torch.zeros_like(tp_tensor)

        return torch.cat([normalized_bw, normalized_tp], dim=1)

    def _load_json_files(self, root: str) -> List[str]:
        """Loads paths to all JSON files in the specified directory."""
        data_dir = os.path.join(root, "data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return json_files

    def _process_single_graph(self, file_path: str) -> torch_geometric.data.Data:
        """Processes a single JSON file into a PyG Data object."""
        with open(file_path, 'r') as f:
            data: Dict[str, Any] = json.load(f)

        # 1. Process Nodes (One-Hot Encoding)
        node_ids = [n['id'] for n in data['nodes']]
        node_types = [n['type'] for n in data['nodes']]
        
        # Create mapping from type string to index
        type_to_index = {t: i for i, t in enumerate(self.NODE_TYPES)}
        
        x_list = []
        for node_type in node_types:
            encoding = [0] * self.NUM_NODE_TYPES
            if node_type in type_to_index:
                idx = type_to_index[node_type]
                encoding[idx] = 1
            x_list.append(encoding)

        x = torch.tensor(x_list, dtype=torch.float32)
        
        # Map node IDs to contiguous indices (0 to N-1) for PyG compatibility
        node_map = {id: i for i, id in enumerate(sorted(node_ids))}

        # 2. Process Edges (Normalized Attributes and Indexing)
        edge_list = []
        bandwidths = []
        throughputs = []
        
        for edge in data['edges']:
            source, target = edge['source'], edge['target']
            bw = float(edge['bandwidth'])
            tp = float(edge['throughput'])

            # Use mapped indices for PyG compatibility
            mapped_source = node_map[source]
            mapped_target = node_map[target]
            
            edge_list.append([mapped_source, mapped_target])
            bandwidths.append(bw)
            throughputs.append(tp)

        # Normalize attributes
        edge_attr = self._normalize_attributes(bandwidths, throughputs)
        
        # 3. Process Targets (y)
        score = float(data['score'])
        bottleneck = int(data['bottleneck'])
        security_flaw = int(data['security_flaw'])

        # Concatenate targets into a single tensor [score, bottleneck, security_flaw]
        y = torch.tensor([score, bottleneck, security_flaw], dtype=torch.float32)

        # Create PyG Data object
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _process_all_data(self):
        """Processes all loaded JSON files and stores them as PyG Data objects."""
        print("Processing dataset...")
        for file_path in self.data_files:
            try:
                data_obj = self._process_single_graph(file_path)
                self.processed_data.append(data_obj)
            except Exception as e:
                print(f"Warning: Could not process {file_path}. Error: {e}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        """Retrieves a single PyG Data object by index."""
        return self.processed_data[idx]