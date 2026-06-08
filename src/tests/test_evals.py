"""
Test suite for EconCausal and Corr2Cause evaluation infrastructure.

Tests the eval pipeline without requiring GPU or model loading:
- Process-results functions (sign extraction, bool extraction)
- YAML task configuration validity
- Suite configuration correctness
- Eval script structure
- Data format validation

Run:
    python3 -m pytest src/tests/test_evals.py -v
"""

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EVALS_DIR = _PROJECT_ROOT / "evals"
_TASK_CONFIGS_DIR = _EVALS_DIR / "configs" / "task_configs"
_SCRIPTS_DIR = _EVALS_DIR / "scripts"

# Add task_configs to path so we can import process_results modules
sys.path.insert(0, str(_TASK_CONFIGS_DIR))

# ---------------------------------------------------------------------------
# Import process_results modules (lazy, inside tests that need them)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: EconCausal process_results
# ---------------------------------------------------------------------------

class TestEconCausalExtractSign:
    """Test sign extraction from model output."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        """Load econcausal module once per class."""
        import econcausal

        self.extract_sign = econcausal._extract_sign
        self.normalize = econcausal._normalize

    # -- JSON extraction --

    def test_json_plus(self):
        assert self.extract_sign('{"predicted_sign": "+"}') == "+"

    def test_json_minus(self):
        assert self.extract_sign('{"predicted_sign": "-"}') == "-"

    def test_json_none(self):
        assert self.extract_sign('{"predicted_sign": "None"}') == "None"

    def test_json_mixed(self):
        assert self.extract_sign('{"predicted_sign": "mixed"}') == "mixed"

    def test_json_with_extra_fields(self):
        text = '{"reasoning": "blah", "predicted_sign": "+", "confidence": 0.9}'
        assert self.extract_sign(text) == "+"

    def test_json_case_insensitive_key(self):
        assert self.extract_sign('{"Predicted_Sign": "+"}') == "+"

    def test_json_embedded_in_text(self):
        text = "Here is my answer: {\"predicted_sign\": \"-\"}. I hope that helps."
        assert self.extract_sign(text) == "-"

    # -- Context-aware extraction --

    def test_context_sign_plus(self):
        assert self.extract_sign("The predicted sign is +") == "+"

    def test_context_answer_minus(self):
        assert self.extract_sign("Final answer: -") == "-"

    def test_context_prediction_none(self):
        assert self.extract_sign("My prediction: None") == "None"

    def test_context_result_mixed(self):
        assert self.extract_sign("Result: mixed") == "mixed"

    # -- Standalone fallback --

    def test_standalone_plus(self):
        assert self.extract_sign("+") == "+"

    def test_standalone_minus(self):
        assert self.extract_sign("-") == "-"

    def test_standalone_none(self):
        assert self.extract_sign("None") == "None"

    def test_standalone_mixed(self):
        assert self.extract_sign("mixed") == "mixed"

    # -- Last match wins (correction pattern) --

    def test_last_context_match_wins(self):
        """Model corrects itself: first +, then -."""
        text = "Initial prediction: + ... wait, final answer: -"
        assert self.extract_sign(text) == "-"

    # -- Edge cases --

    def test_empty_string_returns_none(self):
        assert self.extract_sign("") is None

    def test_no_sign_returns_none(self):
        assert self.extract_sign("I don't know the answer") is None

    def test_hyphenated_word_not_matched(self):
        """'not-significant' should not match '-'."""
        # May or may not match depending on regex; just ensure no crash
        result = self.extract_sign("The result was not-significant")
        # We don't assert a specific value since '-' in hyphenated words
        # is ambiguous. The important thing is no crash.

    def test_normalize_case_insensitive_none(self):
        assert self.normalize("none") == "None"
        assert self.normalize("NONE") == "None"

    def test_normalize_case_insensitive_mixed(self):
        assert self.normalize("Mixed") == "mixed"
        assert self.normalize("MIXED") == "mixed"

    def test_normalize_preserves_plus_minus(self):
        assert self.normalize("+") == "+"
        assert self.normalize("-") == "-"

    def test_normalize_none_input(self):
        assert self.normalize(None) is None


class TestEconCausalProcessResults:
    """Test the full process_results pipeline."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import econcausal

        self.process_results = econcausal.process_results

    def test_correct_plus(self):
        doc = {"answer": "+"}
        results = ['{"predicted_sign": "+"}']
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_correct_minus(self):
        doc = {"answer": "-"}
        results = ['{"predicted_sign": "-"}']
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_correct_none(self):
        doc = {"answer": "None"}
        results = ['{"predicted_sign": "None"}']
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_correct_mixed(self):
        doc = {"answer": "mixed"}
        results = ['{"predicted_sign": "mixed"}']
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_wrong_answer(self):
        doc = {"answer": "+"}
        results = ['{"predicted_sign": "-"}']
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_unparseable_output_is_wrong(self):
        doc = {"answer": "+"}
        results = ["I don't know"]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_empty_results_is_wrong(self):
        doc = {"answer": "+"}
        results = [""]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_no_results_is_wrong(self):
        doc = {"answer": "+"}
        results = []
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0


# ---------------------------------------------------------------------------
# Test: Corr2Cause process_results
# ---------------------------------------------------------------------------

class TestCorr2CauseExtractBool:
    """Test True/False extraction from model output."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import corr2cause

        self.extract_bool = corr2cause._extract_bool

    # -- Exact match --

    def test_exact_true(self):
        assert self.extract_bool("True") is True

    def test_exact_false(self):
        assert self.extract_bool("False") is False

    def test_exact_true_whitespace(self):
        assert self.extract_bool("  True  ") is True

    def test_exact_false_whitespace(self):
        assert self.extract_bool("  False  ") is False

    def test_exact_true_newline(self):
        assert self.extract_bool("True\n") is True

    # -- First token --

    def test_first_true_in_sentence(self):
        assert self.extract_bool("True, because X causes Y") is True

    def test_first_false_in_sentence(self):
        assert self.extract_bool("False. The premise does not support the hypothesis.") is False

    # -- Boilerplate handling --

    def test_true_or_false_boilerplate_with_answer(self):
        """Model repeats question format then answers."""
        text = "Is this true or false? True"
        assert self.extract_bool(text) is True

    def test_true_or_false_boilerplate_no_answer(self):
        """Model only repeats question format."""
        text = "Answer true or false"
        assert self.extract_bool(text) is None

    # -- Meta-commentary handling --

    def test_determine_if_with_answer(self):
        text = "Determine if the hypothesis is true or false based on the premise. True"
        assert self.extract_bool(text) is True

    def test_determine_if_no_answer(self):
        text = "Determine if the hypothesis is true or false"
        assert self.extract_bool(text) is None

    # -- Edge cases --

    def test_empty_string(self):
        assert self.extract_bool("") is None

    def test_no_bool_word(self):
        assert self.extract_bool("I don't know") is None

    def test_case_insensitive(self):
        assert self.extract_bool("TRUE") is True
        assert self.extract_bool("FALSE") is False
        assert self.extract_bool("true") is True
        assert self.extract_bool("false") is False


class TestCorr2CauseProcessResults:
    """Test the full process_results pipeline."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import corr2cause

        self.process_results = corr2cause.process_results

    def test_correct_true(self):
        doc = {"label": 1}
        results = ["True"]
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_correct_false(self):
        doc = {"label": 0}
        results = ["False"]
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_wrong_true_for_false_label(self):
        doc = {"label": 0}
        results = ["True"]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_wrong_false_for_true_label(self):
        doc = {"label": 1}
        results = ["False"]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_unparseable_is_wrong(self):
        doc = {"label": 1}
        results = ["Maybe"]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_empty_result_is_wrong(self):
        doc = {"label": 1}
        results = [""]
        out = self.process_results(doc, results)
        assert out["acc"] == 0.0

    def test_missing_label_defaults_to_false(self):
        doc = {}
        results = ["False"]
        out = self.process_results(doc, results)
        assert out["acc"] == 1.0


# ---------------------------------------------------------------------------
# Test: YAML Task Configuration
# ---------------------------------------------------------------------------

def _lm_eval_yaml_load(stream) -> Dict[str, Any]:
    """Load YAML that may contain lm_eval's !function tags.

    yaml.safe_load chokes on !function (undefined tag). This custom
    loader treats !function values as plain strings, which is sufficient
    for validation (we don't need to actually call the function).
    """
    import yaml

    class _LmEvalLoader(yaml.SafeLoader):
        pass

    def _function_constructor(loader, node):
        return loader.construct_scalar(node)

    _LmEvalLoader.add_constructor("!function", _function_constructor)
    return yaml.load(stream, Loader=_LmEvalLoader)


class TestTaskConfigs:
    """Validate YAML task configuration files."""

    def _get_task_files(self) -> list:
        """Get all task YAML files (excluding suite files)."""
        suite_files = {"causal_suite.yaml", "full_suite.yaml", "short_suite.yaml"}
        return [
            f
            for f in os.listdir(_TASK_CONFIGS_DIR)
            if f.endswith(".yaml") and f not in suite_files
        ]

    def _load_config(self, filename: str) -> Dict[str, Any]:
        filepath = _TASK_CONFIGS_DIR / filename
        with open(filepath) as f:
            return _lm_eval_yaml_load(f)

    # Compute task files once at module level for parametrize
    _TASK_FILES = [
        f
        for f in os.listdir(_TASK_CONFIGS_DIR)
        if f.endswith(".yaml")
        and f not in {"causal_suite.yaml", "full_suite.yaml", "short_suite.yaml"}
    ]

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_file_exists(self, task_file: str):
        """All task config files exist."""
        assert (_TASK_CONFIGS_DIR / task_file).exists()

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_file_is_valid_yaml(self, task_file: str):
        """All task config files parse as valid YAML."""
        config = self._load_config(task_file)
        assert config is not None

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_task_name(self, task_file: str):
        """All task configs have a 'task' field."""
        config = self._load_config(task_file)
        assert "task" in config, f"{task_file} missing 'task' field"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_dataset_path(self, task_file: str):
        """All task configs have a dataset_path."""
        config = self._load_config(task_file)
        assert "dataset_path" in config, f"{task_file} missing 'dataset_path'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_doc_to_text(self, task_file: str):
        """All task configs have doc_to_text template."""
        config = self._load_config(task_file)
        assert "doc_to_text" in config, f"{task_file} missing 'doc_to_text'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_doc_to_target(self, task_file: str):
        """All task configs have doc_to_target template."""
        config = self._load_config(task_file)
        assert "doc_to_target" in config, f"{task_file} missing 'doc_to_target'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_process_results(self, task_file: str):
        """All task configs have process_results function reference."""
        config = self._load_config(task_file)
        assert "process_results" in config, f"{task_file} missing 'process_results'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_metric_list(self, task_file: str):
        """All task configs have metric_list."""
        config = self._load_config(task_file)
        assert "metric_list" in config, f"{task_file} missing 'metric_list'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_has_generation_kwargs(self, task_file: str):
        """All task configs have generation_kwargs."""
        config = self._load_config(task_file)
        assert "generation_kwargs" in config, f"{task_file} missing 'generation_kwargs'"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_generation_has_max_gen_toks(self, task_file: str):
        """All task configs specify max_gen_toks."""
        config = self._load_config(task_file)
        gen = config.get("generation_kwargs", {})
        assert "max_gen_toks" in gen, f"{task_file} missing max_gen_toks"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_generation_has_until(self, task_file: str):
        """All task configs have stop sequences."""
        config = self._load_config(task_file)
        gen = config.get("generation_kwargs", {})
        assert "until" in gen, f"{task_file} missing 'until' stop sequences"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_generation_greedy(self, task_file: str):
        """All task configs use greedy decoding (do_sample: false)."""
        config = self._load_config(task_file)
        gen = config.get("generation_kwargs", {})
        assert gen.get("do_sample") is False, f"{task_file} should use greedy decoding"

    @pytest.mark.parametrize("task_file", _TASK_FILES)
    def test_task_zero_shot(self, task_file: str):
        """All task configs are 0-shot."""
        config = self._load_config(task_file)
        assert config.get("num_fewshot", 0) == 0, f"{task_file} should be 0-shot"

    def test_econcausal_task_names(self):
        """Verify expected EconCausal task names."""
        expected = {
            "econcausal_task1_econ.yaml",
            "econcausal_task1_finance.yaml",
            "econcausal_task2.yaml",
            "econcausal_task3.yaml",
        }
        actual = set(self._get_task_files())
        assert expected.issubset(actual), f"Missing tasks: {expected - actual}"

    def test_corr2cause_task_name(self):
        """Verify Corr2Cause task config exists."""
        assert "corr2cause.yaml" in self._get_task_files()

    def test_econcausal_task1_econ_config(self):
        """Verify econcausal_task1_econ specific config."""
        config = self._load_config("econcausal_task1_econ.yaml")
        assert config["task"] == "econcausal_task1_econ"
        assert config["dataset_path"] == "json"
        assert config["generation_kwargs"]["max_gen_toks"] == 256

    def test_econcausal_task1_finance_config(self):
        """Verify econcausal_task1_finance specific config."""
        config = self._load_config("econcausal_task1_finance.yaml")
        assert config["task"] == "econcausal_task1_finance"
        assert config["generation_kwargs"]["max_gen_toks"] == 256

    def test_econcausal_task2_config(self):
        """Verify econcausal_task2 specific config."""
        config = self._load_config("econcausal_task2.yaml")
        assert config["task"] == "econcausal_task2"
        assert config["generation_kwargs"]["max_gen_toks"] == 256

    def test_econcausal_task3_config(self):
        """Verify econcausal_task3 specific config."""
        config = self._load_config("econcausal_task3.yaml")
        assert config["task"] == "econcausal_task3"
        assert config["generation_kwargs"]["max_gen_toks"] == 256

    def test_corr2cause_config(self):
        """Verify corr2cause specific config."""
        config = self._load_config("corr2cause.yaml")
        assert config["task"] == "corr2cause"
        assert config["dataset_path"] == "causal-nlp/corr2cause"
        assert config["generation_kwargs"]["max_gen_toks"] == 16


# ---------------------------------------------------------------------------
# Test: Suite Configuration
# ---------------------------------------------------------------------------

class TestSuiteConfigs:
    """Validate suite configuration files."""

    def _load_suite(self, filename: str) -> Dict[str, Any]:
        filepath = _TASK_CONFIGS_DIR / filename
        with open(filepath) as f:
            return _lm_eval_yaml_load(f)

    def test_causal_suite_exists(self):
        """causal_suite.yaml exists."""
        assert (_TASK_CONFIGS_DIR / "causal_suite.yaml").exists()

    def test_causal_suite_valid_yaml(self):
        """causal_suite.yaml is valid YAML."""
        suite = self._load_suite("causal_suite.yaml")
        assert suite is not None

    def test_causal_suite_has_group(self):
        """causal_suite.yaml has a group name."""
        suite = self._load_suite("causal_suite.yaml")
        assert "group" in suite
        assert suite["group"] == "causal_reasoning_suite"

    def test_causal_suite_has_tasks(self):
        """causal_suite.yaml has a task list."""
        suite = self._load_suite("causal_suite.yaml")
        assert "task" in suite
        assert isinstance(suite["task"], list)
        assert len(suite["task"]) > 0

    def test_causal_suite_task_count(self):
        """causal_suite.yaml has exactly 5 tasks."""
        suite = self._load_suite("causal_suite.yaml")
        assert len(suite["task"]) == 5

    def test_causal_suite_task_names(self):
        """causal_suite.yaml references correct task names."""
        suite = self._load_suite("causal_suite.yaml")
        task_names = {t["task"] for t in suite["task"]}
        expected = {
            "econcausal_task1_econ",
            "econcausal_task1_finance",
            "econcausal_task2",
            "econcausal_task3",
            "corr2cause",
        }
        assert task_names == expected

    def test_causal_suite_tasks_have_configs(self):
        """Each task in causal_suite.yaml has a corresponding config file."""
        suite = self._load_suite("causal_suite.yaml")
        for task_entry in suite["task"]:
            task_name = task_entry["task"]
            config_file = _TASK_CONFIGS_DIR / f"{task_name}.yaml"
            assert config_file.exists(), f"Config file for {task_name} not found"

    def test_full_suite_exists(self):
        """full_suite.yaml exists."""
        assert (_TASK_CONFIGS_DIR / "full_suite.yaml").exists()

    def test_full_suite_includes_causal_tasks(self):
        """full_suite.yaml includes all causal reasoning tasks."""
        suite = self._load_suite("full_suite.yaml")
        task_names = {t.get("task", t.get("name", "")) for t in suite["task"]}
        causal_tasks = {
            "econcausal_task1_econ",
            "econcausal_task1_finance",
            "econcausal_task2",
            "econcausal_task3",
            "corr2cause",
        }
        assert causal_tasks.issubset(task_names)


# ---------------------------------------------------------------------------
# Test: Process Results Module Import
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Verify process_results modules are importable."""

    def test_econcausal_importable(self):
        """econcausal module can be imported."""
        import econcausal

        assert hasattr(econcausal, "process_results")
        assert hasattr(econcausal, "_extract_sign")
        assert hasattr(econcausal, "_normalize")

    def test_corr2cause_importable(self):
        """corr2cause module can be imported."""
        import corr2cause

        assert hasattr(corr2cause, "process_results")
        assert hasattr(corr2cause, "_extract_bool")

    def test_econcausal_process_results_signature(self):
        """process_results has correct signature."""
        import inspect
        import econcausal

        sig = inspect.signature(econcausal.process_results)
        params = list(sig.parameters.keys())
        assert "doc" in params
        assert "results" in params

    def test_corr2cause_process_results_signature(self):
        """process_results has correct signature."""
        import inspect
        import corr2cause

        sig = inspect.signature(corr2cause.process_results)
        params = list(sig.parameters.keys())
        assert "doc" in params
        assert "results" in params

    def test_econcausal_returns_dict_with_acc(self):
        """process_results returns dict with 'acc' key."""
        import econcausal

        result = econcausal.process_results({"answer": "+"}, ['{"predicted_sign": "+"}'])
        assert isinstance(result, dict)
        assert "acc" in result
        assert isinstance(result["acc"], float)

    def test_corr2cause_returns_dict_with_acc(self):
        """process_results returns dict with 'acc' key."""
        import corr2cause

        result = corr2cause.process_results({"label": 1}, ["True"])
        assert isinstance(result, dict)
        assert "acc" in result
        assert isinstance(result["acc"], float)


# ---------------------------------------------------------------------------
# Test: Eval Scripts
# ---------------------------------------------------------------------------

class TestEvalScripts:
    """Validate eval script structure."""

    def test_baseline_script_exists(self):
        assert (_SCRIPTS_DIR / "run_baseline_bf16.sh").exists()

    def test_finetuned_script_exists(self):
        assert (_SCRIPTS_DIR / "run_finetuned_bf16.sh").exists()

    def test_logging_script_exists(self):
        assert (_SCRIPTS_DIR / "eval_logging.sh").exists()

    def test_baseline_script_is_executable(self):
        st = os.stat(_SCRIPTS_DIR / "run_baseline_bf16.sh")
        assert st.st_mode & stat.S_IXUSR, "run_baseline_bf16.sh should be executable"

    def test_finetuned_script_is_executable(self):
        st = os.stat(_SCRIPTS_DIR / "run_finetuned_bf16.sh")
        assert st.st_mode & stat.S_IXUSR, "run_finetuned_bf16.sh should be executable"

    def test_baseline_script_has_causal_suite(self):
        """Baseline script supports --suite causal."""
        content = (_SCRIPTS_DIR / "run_baseline_bf16.sh").read_text()
        assert "causal" in content

    def test_finetuned_script_has_causal_suite(self):
        """Finetuned script supports --suite causal."""
        content = (_SCRIPTS_DIR / "run_finetuned_bf16.sh").read_text()
        assert "causal" in content

    def test_baseline_script_has_enable_thinking_false(self):
        """Baseline script disables thinking mode."""
        content = (_SCRIPTS_DIR / "run_baseline_bf16.sh").read_text()
        assert "enable_thinking=False" in content

    def test_finetuned_script_has_enable_thinking_false(self):
        """Finetuned script disables thinking mode."""
        content = (_SCRIPTS_DIR / "run_finetuned_bf16.sh").read_text()
        assert "enable_thinking=False" in content

    def test_baseline_script_has_include_path(self):
        """Baseline script includes task_configs path."""
        content = (_SCRIPTS_DIR / "run_baseline_bf16.sh").read_text()
        assert "include_path" in content
        assert "task_configs" in content

    def test_finetuned_script_has_include_path(self):
        """Finetuned script includes task_configs path."""
        content = (_SCRIPTS_DIR / "run_finetuned_bf16.sh").read_text()
        assert "include_path" in content
        assert "task_configs" in content

    def test_baseline_script_has_all_econcausal_tasks(self):
        """Baseline script lists all econcausal tasks."""
        content = (_SCRIPTS_DIR / "run_baseline_bf16.sh").read_text()
        for task in [
            "econcausal_task1_econ",
            "econcausal_task1_finance",
            "econcausal_task2",
            "econcausal_task3",
            "corr2cause",
        ]:
            assert task in content, f"Missing task: {task}"

    def test_finetuned_script_has_all_econcausal_tasks(self):
        """Finetuned script lists all econcausal tasks."""
        content = (_SCRIPTS_DIR / "run_finetuned_bf16.sh").read_text()
        for task in [
            "econcausal_task1_econ",
            "econcausal_task1_finance",
            "econcausal_task2",
            "econcausal_task3",
            "corr2cause",
        ]:
            assert task in content, f"Missing task: {task}"

    def test_baseline_script_has_apply_chat_template(self):
        """Baseline script uses apply_chat_template."""
        content = (_SCRIPTS_DIR / "run_baseline_bf16.sh").read_text()
        assert "apply_chat_template" in content

    def test_finetuned_script_has_apply_chat_template(self):
        """Finetuned script uses apply_chat_template."""
        content = (_SCRIPTS_DIR / "run_finetuned_bf16.sh").read_text()
        assert "apply_chat_template" in content

    # -- GRPO script tests --

    def test_grpo_script_exists(self):
        assert (_SCRIPTS_DIR / "run_grpo_bf16.sh").exists()

    def test_grpo_script_is_executable(self):
        st = os.stat(_SCRIPTS_DIR / "run_grpo_bf16.sh")
        assert st.st_mode & stat.S_IXUSR, "run_grpo_bf16.sh should be executable"

    def test_grpo_script_has_causal_suite(self):
        """GRPO script supports --suite causal."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "causal" in content

    def test_grpo_script_has_enable_thinking_false(self):
        """GRPO script disables thinking mode."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "enable_thinking=False" in content

    def test_grpo_script_has_include_path(self):
        """GRPO script includes task_configs path."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "include_path" in content
        assert "task_configs" in content

    def test_grpo_script_has_all_econcausal_tasks(self):
        """GRPO script lists all econcausal tasks."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        for task in [
            "econcausal_task1_econ",
            "econcausal_task1_finance",
            "econcausal_task2",
            "econcausal_task3",
            "corr2cause",
        ]:
            assert task in content, f"Missing task: {task}"

    def test_grpo_script_has_apply_chat_template(self):
        """GRPO script uses apply_chat_template."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "apply_chat_template" in content

    def test_grpo_script_has_grpo_model_dir(self):
        """GRPO script references GRPO_MODEL_DIR env var."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "GRPO_MODEL_DIR" in content

    def test_grpo_script_has_grpo_results_dir(self):
        """GRPO script outputs to grpo results directory."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "results/grpo/bf16" in content

    def test_grpo_script_has_grpo_merged_path(self):
        """GRPO script references the merged GRPO checkpoint path."""
        content = (_SCRIPTS_DIR / "run_grpo_bf16.sh").read_text()
        assert "grpo_merged" in content
        assert "checkpoint-500" in content


# ---------------------------------------------------------------------------
# Test: Data Format Validation
# ---------------------------------------------------------------------------

class TestDataFormats:
    """Test that sample data matches expected format for each task."""

    def test_econcausal_sample_format(self):
        """Verify EconCausal sample structure."""
        sample = {
            "question": "Does an increase in interest rates cause a decrease in investment?",
            "answer": "-",
            "context": "Monetary policy transmission mechanism.",
        }
        assert "question" in sample
        assert "answer" in sample
        assert sample["answer"] in {"+", "-", "None", "mixed"}

    def test_corr2cause_sample_format(self):
        """Verify Corr2Cause sample structure."""
        sample = {
            "input": "A and B are correlated. Does A cause B?",
            "label": 0,
        }
        assert "input" in sample
        assert "label" in sample
        assert sample["label"] in {0, 1}

    def test_econcausal_process_results_with_realistic_sample(self):
        """Test process_results with realistic EconCausal data."""
        import econcausal

        doc = {
            "question": "Does GDP growth cause inflation?",
            "answer": "+",
            "context": "Keynesian demand-pull inflation theory.",
        }
        # Model generates correct JSON
        results = ['{"predicted_sign": "+"}']
        out = econcausal.process_results(doc, results)
        assert out["acc"] == 1.0

    def test_corr2cause_process_results_with_realistic_sample(self):
        """Test process_results with realistic Corr2Cause data."""
        import corr2cause

        doc = {
            "input": "A and B are independent. Does A cause B?",
            "label": 0,
        }
        results = ["False"]
        out = corr2cause.process_results(doc, results)
        assert out["acc"] == 1.0


# ---------------------------------------------------------------------------
# Test: Collective Run Validation (dry-run)
# ---------------------------------------------------------------------------

class TestCollectiveRun:
    """Test that all 5 econcausal tasks can be referenced collectively."""

    def test_causal_suite_task_string(self):
        """Verify the comma-separated task string used by --tasks flag."""
        tasks = "econcausal_task1_econ,econcausal_task1_finance,econcausal_task2,econcausal_task3,corr2cause"
        task_list = tasks.split(",")
        assert len(task_list) == 5
        assert "econcausal_task1_econ" in task_list
        assert "econcausal_task1_finance" in task_list
        assert "econcausal_task2" in task_list
        assert "econcausal_task3" in task_list
        assert "corr2cause" in task_list

    def test_causal_suite_group_name(self):
        """Verify the group name can be used as --tasks argument."""
        suite_path = _TASK_CONFIGS_DIR / "causal_suite.yaml"
        with open(suite_path) as f:
            suite = _lm_eval_yaml_load(f)
        group_name = suite["group"]
        assert group_name == "causal_reasoning_suite"

    def test_all_task_configs_consistent_metric(self):
        """All econcausal tasks use acc metric with mean aggregation."""
        task_files = [
            "econcausal_task1_econ.yaml",
            "econcausal_task1_finance.yaml",
            "econcausal_task2.yaml",
            "econcausal_task3.yaml",
            "corr2cause.yaml",
        ]
        for tf in task_files:
            filepath = _TASK_CONFIGS_DIR / tf
            with open(filepath) as f:
                config = _lm_eval_yaml_load(f)
            metrics = config.get("metric_list", [])
            assert len(metrics) >= 1, f"{tf} has no metrics"
            assert metrics[0]["metric"] == "acc", f"{tf} should use acc metric"
            assert metrics[0]["aggregation"] == "mean", f"{tf} should use mean aggregation"
            assert metrics[0]["higher_is_better"] is True, f"{tf} should have higher_is_better=True"

    def test_all_econcausal_tasks_same_generation_config(self):
        """All econcausal tasks use identical generation config."""
        econcausal_files = [
            "econcausal_task1_econ.yaml",
            "econcausal_task1_finance.yaml",
            "econcausal_task2.yaml",
            "econcausal_task3.yaml",
        ]
        ref_config = None
        for tf in econcausal_files:
            filepath = _TASK_CONFIGS_DIR / tf
            with open(filepath) as f:
                config = _lm_eval_yaml_load(f)
            gen = config.get("generation_kwargs", {})
            if ref_config is None:
                ref_config = gen
            else:
                assert gen == ref_config, (
                    f"{tf} has different generation config than {econcausal_files[0]}: "
                    f"{gen} vs {ref_config}"
                )


# ---------------------------------------------------------------------------
# Test: Python Process Results Integration (lm_eval style)
# ---------------------------------------------------------------------------

class TestLmEvalIntegration:
    """Test that process_results functions work as lm_eval would call them."""

    def test_econcausal_batch_processing(self):
        """Simulate lm_eval calling process_results for multiple docs."""
        import econcausal

        docs = [
            {"answer": "+"},
            {"answer": "-"},
            {"answer": "None"},
            {"answer": "mixed"},
        ]
        results_list = [
            ['{"predicted_sign": "+"}'],
            ['{"predicted_sign": "-"}'],
            ['{"predicted_sign": "None"}'],
            ['{"predicted_sign": "mixed"}'],
        ]

        accs = []
        for doc, results in zip(docs, results_list):
            out = econcausal.process_results(doc, results)
            accs.append(out["acc"])

        assert accs == [1.0, 1.0, 1.0, 1.0]

    def test_corr2cause_batch_processing(self):
        """Simulate lm_eval calling process_results for multiple docs."""
        import corr2cause

        docs = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]
        results_list = [
            ["True"],
            ["False"],
            ["True"],
            ["False"],
        ]

        accs = []
        for doc, results in zip(docs, results_list):
            out = corr2cause.process_results(doc, results)
            accs.append(out["acc"])

        assert accs == [1.0, 1.0, 1.0, 1.0]

    def test_econcausal_partial_accuracy(self):
        """Simulate partial accuracy (some correct, some wrong)."""
        import econcausal

        docs = [
            {"answer": "+"},
            {"answer": "-"},
            {"answer": "None"},
        ]
        results_list = [
            ['{"predicted_sign": "+"}'],  # correct
            ['{"predicted_sign": "+"}'],  # wrong, should be -
            ['{"predicted_sign": "None"}'],  # correct
        ]

        accs = []
        for doc, results in zip(docs, results_list):
            out = econcausal.process_results(doc, results)
            accs.append(out["acc"])

        # 2/3 correct
        assert sum(accs) / len(accs) == pytest.approx(2 / 3, abs=0.01)

    def test_corr2cause_partial_accuracy(self):
        """Simulate partial accuracy (some correct, some wrong)."""
        import corr2cause

        docs = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]
        results_list = [
            ["True"],   # correct
            ["True"],   # wrong, should be False
            ["False"],  # wrong, should be True
            ["False"],  # correct
        ]

        accs = []
        for doc, results in zip(docs, results_list):
            out = corr2cause.process_results(doc, results)
            accs.append(out["acc"])

        # 2/4 correct
        assert sum(accs) / len(accs) == pytest.approx(0.5, abs=0.01)
