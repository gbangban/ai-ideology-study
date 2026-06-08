import subprocess

import pytest


class TestSmokeTestModule:
    def test_smoke_test_module_imports(self):
        from src.student import smoke_test
        assert callable(smoke_test.smoke_test)
        assert callable(smoke_test.main)

    def test_cli_help(self):
        result = subprocess.run(
            ["python3", "-m", "src.student.smoke_test", "--help"],
            capture_output=True, text=True,
        )
        if "numpy.dtype size changed" in result.stderr:
            pytest.skip("Host numpy/pandas binary incompatibility")
        assert result.returncode == 0
        assert "--track" in result.stdout
        assert "--num-prompts" in result.stdout

    def test_track_validation_rejects_invalid(self):
        from src.student.smoke_test import _validate_track
        with pytest.raises(ValueError, match="track must be 'outcome' or 'process'"):
            _validate_track("invalid")

    def test_track_validation_accepts_outcome(self):
        from src.student.smoke_test import _validate_track
        assert _validate_track("outcome") == "outcome"

    def test_track_validation_accepts_process(self):
        from src.student.smoke_test import _validate_track
        assert _validate_track("process") == "process"
