import pytest


class TestCausalGraphGenerator:
    def test_dag_generation_no_cycles(self):
        from src.teacher.generate_causal_graphs import generate_random_dag
        import networkx as nx
        dag = generate_random_dag(n_nodes=5, seed=42)
        assert nx.is_directed_acyclic_graph(dag)
        assert dag.number_of_nodes() <= 5

    def test_conditional_independence_query_correct(self):
        from src.teacher.generate_causal_graphs import check_conditional_independence
        import networkx as nx
        dag2 = nx.DiGraph()
        dag2.add_edges_from([(0, 2), (1, 2)])
        result = check_conditional_independence(dag2, 0, 1, conditioning=set())
        assert result is True

    def test_query_generation_format(self):
        from src.teacher.generate_causal_graphs import generate_causal_query
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2), (0, 2)])
        query = generate_causal_query(dag, ["A", "B", "C"], question_idx=0)
        assert "prompt" in query
        assert "answer" in query
        assert query["answer"] in [True, False]

    def test_economic_variable_mapping(self):
        from src.teacher.generate_causal_graphs import get_economic_variables
        vars = get_economic_variables(count=4)
        assert len(vars) == 4

    def test_full_generation_run(self):
        from src.teacher.generate_causal_graphs import generate_causal_graph_dataset
        dataset = generate_causal_graph_dataset(n_samples=10, mode="general", seed=42)
        assert len(dataset) >= 8
        for item in dataset:
            assert "prompt" in item
            assert "answer" in item
            assert item["category"] == "causal_graph"


class TestContextFlipGenerator:
    def test_pair_generation_format(self):
        from src.teacher.generate_context_flips import generate_context_flip_pair
        pair = generate_context_flip_pair(seed=42)
        assert "prompt_a" in pair and "prompt_b" in pair
        assert pair["category"] == "context_flip"
        assert pair["context_a"] != pair["context_b"]

    def test_sufficient_pairs(self):
        from src.teacher.generate_context_flips import generate_context_flip_pairs
        pairs = generate_context_flip_pairs(n_pairs=20, seed=42)
        assert len(pairs) == 20


class TestNullEffectGenerator:
    def test_null_pair_generation(self):
        from src.teacher.generate_null_effects import generate_null_effect_prompt
        prompt = generate_null_effect_prompt(mode="economic", seed=42)
        assert prompt["answer"] == "null"
        assert prompt["category"] == "null_effect"

    def test_both_modes(self):
        from src.teacher.generate_null_effects import generate_null_effect_dataset
        econ = generate_null_effect_dataset(n_samples=5, mode="economic", seed=42)
        gen = generate_null_effect_dataset(n_samples=5, mode="general", seed=42)
        assert len(econ) == 5 and len(gen) == 5


class TestContradictionPairGenerator:
    def test_claim_format(self):
        from src.teacher.generate_contradiction_pairs import generate_contradiction_prompt
        prompt = generate_contradiction_prompt(seed=42)
        assert "prompt" in prompt and "claim" in prompt
        assert prompt["category"] == "contradiction_pair"

    def test_dataset_generation(self):
        from src.teacher.generate_contradiction_pairs import generate_contradiction_dataset
        dataset = generate_contradiction_dataset(n_samples=5, mode="economic", seed=42)
        assert len(dataset) == 5


class TestDatasetAssembler:
    def test_assemble_mixed_dataset(self):
        from src.teacher.build_grpo_dataset import assemble_grpo_dataset
        dataset = assemble_grpo_dataset(
            causal_graph_general=10, causal_graph_economic=20,
            context_flips=15, null_effect_economic=10,
            null_effect_general=5, contradiction_economic=20, seed=42,
        )
        assert len(dataset) >= 70
        categories = [item["category"] for item in dataset]
        assert "causal_graph" in categories and "context_flip" in categories

    def test_output_format(self):
        from src.teacher.build_grpo_dataset import assemble_grpo_dataset
        dataset = assemble_grpo_dataset(
            causal_graph_general=5, causal_graph_economic=5,
            context_flips=5, null_effect_economic=5,
            null_effect_general=5, contradiction_economic=5, seed=42,
        )
        for item in dataset:
            assert "prompt" in item and "category" in item
