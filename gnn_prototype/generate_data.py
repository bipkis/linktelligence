import networkx as nx
import numpy as np
import json
import os
from typing import List, Dict, Any

# --- Configuration ---
NUM_SAMPLES = 500
NODE_TYPES = ["router", "switch", "printer", "host", "firewall", "ids_ips", "l3_switch"]
MIN_BANDWIDTH = 100
MAX_BANDWIDTH = 10000
MIN_THROUGHPUT = 10
MAX_THROUGHPUT = 5000

def generate_random_graph(num_nodes: int) -> nx.Graph:
    """Generates a random connected graph with specified number of nodes."""
    # Use a combination of random edges and ensuring connectivity (e.g., using a path or cycle base)
    G = nx.gnm_random_graph(num_nodes, m=max(num_nodes - 1, num_nodes * 2), seed=np.random.randint(0, 1000))
    # Ensure connectivity by adding edges if necessary (though gnm_random_graph usually handles this well enough for a prototype)
    if not nx.is_connected(G):
        # Simple way to ensure connectivity: add edges between components until connected
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            nodes1 = list(components[i])
            nodes2 = list(components[i+1])
            # Connect a random node from component i to a random node from component i+1
            u = np.random.choice(nodes1)
            v = np.random.choice(nodes2)
            G.add_edge(u, v)
    return G

def assign_attributes(G: nx.Graph) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assigns random types and edge attributes to the graph."""
    nodes = list(G.nodes())
    node_data = []
    for node in nodes:
        # Assign random type
        node_type = np.random.choice(NODE_TYPES)
        node_data.append({"id": node, "type": node_type})

    edges = list(G.edges())
    edge_data = []
    for u, v in edges:
        # Assign random bandwidth and throughput
        bandwidth = np.random.randint(MIN_BANDWIDTH, MAX_BANDWIDTH + 1)
        throughput = np.random.randint(MIN_THROUGHPUT, MAX_THROUGHPUT + 1)
        edge_data.append({"source": u, "target": v, "bandwidth": bandwidth, "throughput": throughput})

    return node_data, edge_data

def calculate_heuristics(G: nx.Graph, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates score, bottleneck, and security_flaw based on heuristic rules."""
    score = 1.0
    bottleneck = 0
    security_flaw = 0

    # Helper mapping for quick type lookup (using node ID as key)
    node_types = {data['id']: data['type'] for data in nodes}
    
    # --- Heuristic 1: Bottleneck Check ---
    firewall_nodes = [n for n, t in node_types.items() if t in ["firewall", "router"]]
    
    critical_node_throughput = {}
    incident_bandwidths = {}

    for node in firewall_nodes:
        total_t = 0
        incident_bw = 0
        # Iterate over neighbors to calculate total throughput and incident bandwidth for this critical node
        for neighbor in G.neighbors(node):
            # Find the edge data for (node, neighbor) pair
            edge_data_lookup = next((e for e in edges if ((e['source'] == node and e['target'] == neighbor') or (e['source'] == neighbor and e['target'] == node))), None)
            if edge_data_lookup:
                total_t += edge_data_lookup['throughput']
                incident_bw += edge_data_lookup['bandwidth']
        critical_node_throughput[node] = total_t
        incident_bandwidths[node] = incident_bw

    for node, total_t in critical_node_throughput.items():
        # Check if the sum of throughput exceeds the combined bandwidth capacity (sum of all incident edge bandwidths)
        if total_t > incident_bandwidths[node]:
            bottleneck = 1
            score -= 0.3

    # --- Heuristic 2: Security Flaw Check ---
    has_host = any(t == "host" for _, t in node_types.items())
    has_firewall = any(t == "firewall" for _, t in node_types.items())
    has_ids_ips = any(t == "ids_ips" for _, t in node_types.items())

    if has_host and not has_firewall and not has_ids_ips:
        security_flaw = 1
        score -= 0.4

    # --- Heuristic 3: Printer to Router Direct Connection Check ---
    for u, v in G.edges():
        type_u = node_types.get(u)
        type_v = node_types.get(v)
        
        is_printer_to_router = (
            (type_u == "printer" and type_v == "router") or 
            (type_u == "router" and type_v == "printer") # Check both directions
        )
        if is_printer_to_router:
            security_flaw += 0.2

    return {"score": score, "bottleneck": bottleneck, "security_flaw": security_flaw}


def main():
    """Main function to generate network data samples."""
    print(f"Generating {NUM_SAMPLES} random network graphs...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for i in range(NUM_SAMPLES):
        # 1. Generate Graph (using a fixed seed for reproducibility of the graph structure per sample)
        num_nodes = np.random.randint(5, 20) # Vary node count slightly
        G = generate_random_graph(num_nodes)

        # 2. Assign Attributes
        node_data, edge_data = assign_attributes(G)

        # 3. Calculate Heuristics
        heuristics = calculate_heuristics(G, node_data, edge_data)

        # 4. Compile final record
        sample_record = {
            "sample_id": i + 1,
            "nodes": node_data,
            "edges": edge_data,
            "metadata": {
                "score": round(heuristics["score"], 3),
                "bottleneck": heuristics["bottleneck"],
                "security_flaw": heuristics["security_flaw"]
            }
        }

        # 5. Save to JSON file
        file_path = os.path.join(data_dir, f"sample_{i+1:04d}.json")
        with open(file_path, 'w') as f:
            json.dump(sample_record, f, indent=4)

    print("Data generation complete. Files saved to the 'data/' directory.")


if __name__ == "__main__":
    main()