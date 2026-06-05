#!/usr/bin/env python3
"""
Programmatic Causal Graph Generator

Generates random DAGs and derives conditional independence queries.
Ground truth is the graph structure, not teacher model output.
"""

import random
import textwrap
from typing import Dict, List, Optional, Set

import networkx as nx


ECONOMIC_VARIABLES = [
    "GDP growth", "Unemployment rate", "Inflation rate", "Interest rates",
    "Government spending", "Tax revenue", "Trade balance", "Wage growth",
    "Consumer confidence", "Housing prices", "Stock market returns",
    "Bank lending", "Corporate profits", "Labor productivity",
    "Money supply", "Exchange rates", "Public debt", "Savings rate",
    "Investment rate", "Consumption spending", "Export volume",
    "Import volume", "Foreign direct investment", "Capital inflow",
    "Credit growth", "Household debt", "Business investment",
    "Real estate investment", "Commodity prices", "Energy prices",
    "Financial regulation strictness", "Labor union density",
    "Minimum wage level", "Corporate tax rate", "Income inequality",
    "Poverty rate", "Social welfare spending", "Education spending",
    "Healthcare spending", "Infrastructure investment",
]

GENERAL_VARIABLES = [
    "Variable A", "Variable B", "Variable C", "Variable D",
    "Variable E", "Variable F", "Variable G", "Variable H",
    "Variable I", "Variable J", "Variable K", "Variable L",
    "Variable M", "Variable N", "Variable O", "Variable P",
]


def get_economic_variables(count: int) -> List[str]:
    return random.sample(ECONOMIC_VARIABLES, min(count, len(ECONOMIC_VARIABLES)))


def get_general_variables(count: int) -> List[str]:
    return random.sample(GENERAL_VARIABLES, min(count, len(GENERAL_VARIABLES)))


def generate_random_dag(
    n_nodes: int = 5,
    edge_prob: float = 0.3,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    if seed is not None:
        random.seed(seed)
    dag = nx.DiGraph()
    nodes = list(range(n_nodes))
    dag.add_nodes_from(nodes)
    topo_order = list(range(n_nodes))
    for i, u in enumerate(topo_order):
        for j, v in enumerate(topo_order):
            if i < j and random.random() < edge_prob:
                dag.add_edge(u, v)
    nodes_with_edges = set()
    for u, v in dag.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)
    isolated = [node for node in dag.nodes() if node not in nodes_with_edges]
    dag.remove_nodes_from(isolated)
    return dag


def check_conditional_independence(
    dag: nx.DiGraph,
    node_a: int,
    node_b: int,
    conditioning: Optional[Set[int]] = None,
) -> bool:
    if conditioning is None:
        conditioning = set()
    return _d_separation_fallback(dag, node_a, node_b, conditioning)


def _d_separation_fallback(
    dag: nx.DiGraph,
    node_a: int,
    node_b: int,
    conditioning: Set[int],
) -> bool:
    all_paths = list(nx.all_simple_paths(dag.to_undirected(), node_a, node_b))
    if not all_paths:
        return True
    for path in all_paths:
        if _path_is_active(dag, path, conditioning):
            return False
    return True


def _path_is_active(
    dag: nx.DiGraph,
    path: List[int],
    conditioning: Set[int],
) -> bool:
    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        node = path[i]
        next_node = path[i + 1]
        arrow_into_from_prev = dag.has_edge(prev_node, node)
        arrow_into_from_next = dag.has_edge(next_node, node)
        is_collider = arrow_into_from_prev and arrow_into_from_next
        if not is_collider and node in conditioning:
            return False
        if is_collider and not _collider_descendant_conditioned(dag, node, conditioning):
            return False
    return True


def _collider_descendant_conditioned(
    dag: nx.DiGraph,
    collider: int,
    conditioning: Set[int],
) -> bool:
    if collider in conditioning:
        return True
    descendants = set(nx.descendants(dag, collider))
    for d in descendants:
        if d in conditioning:
            return True
    return False





def generate_causal_query(
    dag: nx.DiGraph,
    variable_names: List[str],
    question_idx: int = 0,
) -> Optional[Dict]:
    nodes = list(dag.nodes())
    if len(nodes) < 3:
        return None

    relationships = []
    for u, v in dag.edges():
        relationships.append(f"{variable_names[u]} is correlated with {variable_names[v]}")

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not dag.has_edge(nodes[i], nodes[j]) and not dag.has_edge(nodes[j], nodes[i]):
                neighbors_i = set(dag.successors(nodes[i])) | set(dag.predecessors(nodes[i]))
                neighbors_j = set(dag.successors(nodes[j])) | set(dag.predecessors(nodes[j]))
                common = neighbors_i & neighbors_j
                if common:
                    cond_node = list(common)[0]
                    is_indep = check_conditional_independence(dag, nodes[i], nodes[j], conditioning={cond_node})
                    status = "independent" if is_indep else "dependent"
                    relationships.append(
                        f"{variable_names[nodes[i]]} is {status} of {variable_names[nodes[j]]} given {variable_names[cond_node]}"
                    )

    edges = list(dag.edges())
    non_edges = [(i, j) for i in nodes for j in nodes if i != j and not dag.has_edge(i, j)]

    total_queries = len(edges) + min(len(non_edges), len(edges))
    if question_idx < len(edges):
        hypo_a, hypo_b = edges[question_idx]
        answer = True
    elif question_idx < total_queries:
        ne_idx = question_idx - len(edges)
        hypo_a, hypo_b = non_edges[ne_idx % len(non_edges)]
        answer = False
    else:
        return None

    hypothesis = f"{variable_names[hypo_a]} directly causes {variable_names[hypo_b]}"

    prompt = textwrap.dedent(f"""
    Consider the following observed relationships between variables:
    {chr(10).join('- ' + r for r in relationships)}

    Hypothesis: {hypothesis}.
    Is this hypothesis supported by the evidence? Answer True or False.
    """).strip()

    return {
        "prompt": prompt,
        "answer": answer,
        "variables": variable_names,
        "hypothesis": hypothesis,
        "dag_edges": list(dag.edges()),
    }


def generate_causal_graph_dataset(
    n_samples: int = 500,
    mode: str = "general",
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)
    dataset = []
    samples_per_graph = 4
    n_graphs = max(1, n_samples // samples_per_graph)

    for g in range(n_graphs):
        dag = generate_random_dag(n_nodes=random.randint(4, 7), edge_prob=0.3, seed=seed + g)
        nodes = list(dag.nodes())

        if mode == "economic":
            var_names = get_economic_variables(count=len(nodes))
        else:
            var_names = get_general_variables(count=len(nodes))

        remapped = nx.DiGraph()
        idx_map = {old: new for new, old in enumerate(nodes)}
        for u, v in dag.edges():
            remapped.add_edge(idx_map[u], idx_map[v])

        edges = list(remapped.edges())
        non_edges = [(i, j) for i in range(len(nodes)) for j in range(len(nodes)) if i != j and not remapped.has_edge(i, j)]
        total_queries = len(edges) + min(len(non_edges), len(edges))

        for q in range(samples_per_graph):
            if q < total_queries:
                query = generate_causal_query(remapped, var_names, question_idx=q)
                if query:
                    query["category"] = "causal_graph"
                    query["subcategory"] = mode
                    dataset.append(query)
            if len(dataset) >= n_samples:
                break
        if len(dataset) >= n_samples:
            break

    return dataset[:n_samples]


def main():
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(description="Generate causal graph training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--mode", choices=["general", "economic"], default="general")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_causal_graph_dataset(n_samples=args.n_samples, mode=args.mode, seed=args.seed)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(dataset)} causal graph questions -> {args.output}")


if __name__ == "__main__":
    main()
