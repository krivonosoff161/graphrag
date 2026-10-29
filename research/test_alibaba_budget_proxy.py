"""Offline synthetic controls; the fake transport makes no provider calls."""

import contextlib
import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from alibaba_budget_proxy import (
    BudgetProxy, CHAT_PATH, EMBED_PATH, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES,
    MAX_REQUESTS, Rejected,
)


def chat(**changes):
    data = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": "synthetic"}]}
    data.update(changes)
    return json.dumps(data).encode()


class ProxyTests(unittest.TestCase):
    def test_concurrent_failures_are_counted_and_capped(self):
        lock = threading.Lock()
        calls = []

        def failing_transport(path, body):
            with lock:
                calls.append((path, body))
            raise RuntimeError("synthetic provider failure with private marker")

        proxy = BudgetProxy(failing_transport)
        output = io.StringIO()

        def attempt(_):
            try:
                proxy.handle(CHAT_PATH, chat())
            except Rejected as error:
                return error.status, error.message
            self.fail("fake provider failure should be rejected")

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with ThreadPoolExecutor(max_workers=24) as pool:
                results = list(pool.map(attempt, range(200)))
        self.assertEqual(len(calls), MAX_REQUESTS)
        self.assertEqual(proxy.attempted, MAX_REQUESTS)
        self.assertEqual(sum(status == 502 for status, _ in results), MAX_REQUESTS)
        self.assertEqual(sum(status == 429 for status, _ in results), 160)
        self.assertNotIn("private marker", repr(results) + output.getvalue())
        self.assertEqual(output.getvalue(), "")

    def test_rejects_unexpected_paths_models_and_unbounded_requests(self):
        calls = []
        proxy = BudgetProxy(lambda path, body: calls.append((path, body)) or b"{}")
        invalid = [
            ("/v1/models", chat()),
            (CHAT_PATH + "?extra=1", chat()),
            (CHAT_PATH, chat(model="other")),
            (CHAT_PATH, chat(max_tokens=2049)),
            (CHAT_PATH, chat(stream="yes")),
            (CHAT_PATH, chat(response_format={"type": "xml"})),
            (CHAT_PATH, chat(messages=[{"role": "user", "content": [{"text": "x"}]}])),
            (CHAT_PATH, b'{"model":"qwen3.8-flash","model":"other"}'),
            (CHAT_PATH, chat() + b" " * MAX_INPUT_BYTES),
            (EMBED_PATH, json.dumps({"model": "qwen3.8-flash", "input": "x"}).encode()),
            (EMBED_PATH, json.dumps({"model": "text-embedding-v4", "input": "x", "dimensions": 2048}).encode()),
        ]
        for path, body in invalid:
            with self.subTest(path=path, body_length=len(body)), self.assertRaises(Rejected):
                proxy.handle(path, body)
        self.assertEqual(calls, [])
        self.assertEqual(proxy.attempted, 0)

    def test_bounded_output_and_authorized_models(self):
        calls = []
        def fake(path, body):
            calls.append((path, json.loads(body)))
            return b"{}" if len(calls) == 1 else b"x" * (MAX_OUTPUT_BYTES + 1)
        proxy = BudgetProxy(fake)
        self.assertEqual(proxy.handle(CHAT_PATH, chat()), b"{}")
        self.assertEqual(calls[0][1]["max_tokens"], 2048)
        embedding = json.dumps({"model": "text-embedding-v4", "input": ["fictional"]}).encode()
        with self.assertRaises(Rejected) as raised:
            proxy.handle(EMBED_PATH, embedding)
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(proxy.attempted, 2)

    def test_graphrag_compatible_stream_schema_and_embedding(self):
        calls = []
        def fake(path, body):
            calls.append((path, json.loads(body)))
            return b'data: {"choices":[]}\n\ndata: [DONE]\n\n' if path == CHAT_PATH else b'{"data":[]}'
        proxy = BudgetProxy(fake)
        schema = {"type": "json_schema", "json_schema": {"name": "Report", "schema": {"type": "object"}, "strict": True}}
        result = proxy.handle(CHAT_PATH, chat(stream=True, response_format=schema))
        self.assertIn(b"data: [DONE]", result)
        embedding = json.dumps({"model": "text-embedding-v4", "input": "fictional", "dimensions": 1024}).encode()
        self.assertEqual(proxy.handle(EMBED_PATH, embedding), b'{"data":[]}')
        self.assertTrue(calls[0][1]["stream"])
        self.assertEqual(calls[1][1]["dimensions"], 1024)


if __name__ == "__main__":
    unittest.main()
