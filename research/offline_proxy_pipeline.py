"""Exercise the real GraphRAG client against a synthetic local transport."""
import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from alibaba_budget_proxy import BudgetProxy, CHAT_PATH, make_handler


def fake_transport(path, body):
    request = json.loads(body)
    if path != CHAT_PATH:
        values = request["input"]
        if isinstance(values, str):
            values = [values]
        return json.dumps({"object": "list", "model": "text-embedding-v4", "data": [
            {"object": "embedding", "index": i, "embedding": [0.01] * 1024}
            for i in range(len(values))], "usage": {"prompt_tokens": 1, "total_tokens": 1}}).encode()
    message = '("entity"<|>ALICE<|>PERSON<|>A fictional person)##("entity"<|>TEAM<|>ORGANIZATION<|>A fictional team)##("relationship"<|>ALICE<|>TEAM<|>Alice works with Team<|>5)##<|COMPLETE|>'
    if "response_format" in request:
        message = json.dumps({"title": "Fictional team", "summary": "Synthetic",
                              "rating": 1.0, "rating_explanation": "Synthetic",
                              "findings": [{"summary": "Synthetic", "explanation": "Synthetic"}]})
    if request.get("stream"):
        chunk = {"id": "fake", "object": "chat.completion.chunk", "created": 1,
                 "model": "qwen3.8-flash", "choices": [{"index": 0, "delta": {"content": "Fictional answer"}, "finish_reason": None}]}
        end = {"id": "fake", "object": "chat.completion.chunk", "created": 1,
               "model": "qwen3.8-flash", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        return ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode()
    return json.dumps({"id": "fake", "object": "chat.completion", "created": 1, "model": "qwen3.8-flash",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": message}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), make_handler(BudgetProxy(fake_transport)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env["GRAPH_RESEARCH_PROXY_BASE"] = "http://127.0.0.1:8765/v1"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("real_model_pipeline.py"))], env=env, timeout=600)
        if result.returncode:
            raise RuntimeError("offline integration failed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
