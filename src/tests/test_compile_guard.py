import torch
import torch.nn as nn

from src.student.train_grpo_base import CompileGuardedModel, maybe_compile_model


class TestCompileGuardedModel:
    def test_basic_forward_pass(self):
        model = nn.Linear(10, 5)
        guarded = CompileGuardedModel(model)

        x = torch.randn(2, 10)
        out = guarded.forward(x)
        assert out.shape == (2, 5)

    def test_compiled_property_is_boolean(self):
        model = nn.Linear(10, 5)
        guarded = CompileGuardedModel(model)

        assert isinstance(guarded.compiled, bool)

    def test_maybe_compile_disabled(self):
        model = nn.Linear(10, 5)
        result, was_compiled = maybe_compile_model(model, enable=False)

        assert result is model
        assert was_compiled is False

    def test_maybe_compile_enabled(self):
        model = nn.Linear(10, 5)
        result, was_compiled = maybe_compile_model(model, enable=True)

        assert isinstance(result, CompileGuardedModel)
        assert isinstance(was_compiled, bool)

        x = torch.randn(2, 10)
        out = result.forward(x)
        assert out.shape == (2, 5)
