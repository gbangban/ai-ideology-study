import pytest
from unittest.mock import patch, MagicMock
import json


class TestSglangClientHealth:
    def test_health_check_returns_true_when_reachable(self):
        from src.student.legacy.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "test", "object": "model"}]}

        with patch("requests.get", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            assert client.health_check() is True

    def test_health_check_returns_false_when_unreachable(self):
        from src.student.legacy.sglang_client import SglangClient

        with patch("requests.get", side_effect=Exception("Connection refused")):
            client = SglangClient("http://localhost:1235")
            assert client.health_check() is False


class TestSglangClientChatCompletion:
    def test_chat_completion_returns_content(self):
        from src.student.legacy.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "STRUCTURAL_ANALYSIS: Yes\nCONTRADICTION_TRACING: Yes\nFRAME_CRITIQUE: No\nCONCLUSION_DIVERGENCE: Yes"
                    }
                }
            ]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client = SglangClient("http://localhost:1235")
            result = client.chat_completion([{"role": "user", "content": "test"}])
            assert "STRUCTURAL_ANALYSIS" in result
            mock_post.assert_called_once()

    def test_chat_completion_raises_on_http_error(self):
        from src.student.legacy.sglang_client import SglangClient
        import requests as req

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            with pytest.raises(req.HTTPError):
                client.chat_completion([{"role": "user", "content": "test"}])


class TestSglangClientBatchCompletion:
    def test_batch_chat_completion_returns_all_results(self):
        from src.student.legacy.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }

        with patch("requests.post", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            requests_list = [
                {"messages": [{"role": "user", "content": f"q{i}"}]}
                for i in range(4)
            ]
            results = client.batch_chat_completion(requests_list)
            assert len(results) == 4
            assert all(r == "response" for r in results)

    def test_batch_chat_completion_retries_on_failure(self):
        from src.student.legacy.sglang_client import SglangClient
        import requests as req

        call_count = [0]
        def mock_post_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise req.HTTPError("503 Service Unavailable")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        with patch("requests.post", side_effect=mock_post_side_effect):
            client = SglangClient("http://localhost:1235", timeout=1)
            results = client.batch_chat_completion([
                {"messages": [{"role": "user", "content": "test"}]}
            ])
            assert len(results) == 1
            assert results[0] == "ok"
            assert call_count[0] == 3
