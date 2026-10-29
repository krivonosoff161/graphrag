"""Run one synthetic GraphRAG experiment with an independently bounded proxy."""
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    token = os.environ.pop("GRAPH_RESEARCH_TEMP_TOKEN", "")
    if not token.startswith("st-"):
        raise RuntimeError("temporary token unavailable")
    proxy_env = os.environ.copy()
    proxy_env["DASHSCOPE_API_KEY"] = token
    child_env = os.environ.copy()
    child_env["GRAPH_RESEARCH_PROXY_BASE"] = "http://127.0.0.1:8765/v1"
    research = Path(__file__).resolve().parent
    proxy = subprocess.Popen([sys.executable, str(research / "alibaba_budget_proxy.py")],
                             env=proxy_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    del proxy_env["DASHSCOPE_API_KEY"]
    token = None
    try:
        ready = False
        for _ in range(30):
            if proxy.poll() is not None:
                raise RuntimeError("proxy did not start")
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/", timeout=1)
            except urllib.error.HTTPError as error:
                if error.code == 405:
                    ready = True
                    break
            except urllib.error.URLError:
                pass
            time.sleep(0.1)
        if not ready:
            raise RuntimeError("proxy readiness failed")
        result = subprocess.run([sys.executable, str(research / "real_model_pipeline.py")],
                                env=child_env, timeout=900)
        if result.returncode:
            raise RuntimeError("synthetic GraphRAG pipeline failed")
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait(timeout=5)


if __name__ == "__main__":
    main()
