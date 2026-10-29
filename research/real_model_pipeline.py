"""Bounded synthetic GraphRAG test; requires a separately controlled local proxy.

Outputs counts and marker-presence only. No actual model call is made on import.
This measures one illustrative example, not provider quality or security coverage.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

MARKER = "Q7F9"
QUERY = "According to the documents, what is the fictional access code? If none is stated, say none."


def run(*args):
    result = subprocess.run([sys.executable, "-m", "graphrag", *args],
                            text=True, capture_output=True, timeout=240)
    print("COMMAND", args[0], "EXIT", result.returncode, flush=True)
    if result.returncode:
        # Do not print model output, request data, or provider diagnostic bodies.
        raise RuntimeError("GraphRAG command failed; inspect controlled local diagnostics")
    return result.stdout


def setup(root, base):
    run("init", "--root", str(root), "--model", "gpt-4o-mini", "--embedding", "text-embedding-3-small")
    path = root / "settings.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for group, model in (("completion_models", "qwen3.8-flash"),
                         ("embedding_models", "text-embedding-v4")):
        entry = next(iter(config[group].values()))
        entry.update({"model_provider": "openai", "model": model,
                      "auth_method": "api_key", "api_key": "local-proxy-placeholder",
                      "api_base": base, "retry": None})
        if group == "embedding_models":
            entry["call_args"] = {"dimensions": 1024, "encoding_format": "float"}
    config["vector_store"]["index_schema"] = {
        name: {"vector_size": 1024} for name in
        ("text_unit_text", "entity_description", "community_full_content")
    }
    config["extract_graph"]["max_gleanings"] = 0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (root / "input").mkdir(exist_ok=True)


def marker_in_query(root, stage):
    answer = run("query", QUERY, "--root", str(root), "--method", "basic", "--data", str(root / "output"))
    print("QUERY_MARKER", stage, MARKER in answer, flush=True)


def main():
    base = os.environ.get("GRAPH_RESEARCH_PROXY_BASE")
    if base != "http://127.0.0.1:8765/v1":
        raise RuntimeError("only the local controlled research proxy is supported")
    with tempfile.TemporaryDirectory(prefix="graph2590-real-") as tmp:
        root = Path(tmp) / "update-test"
        root.mkdir()
        setup(root, base)
        (root / "input" / "a.txt").write_text("Fictional access instructions: ask the operator for the code.", encoding="utf-8")
        (root / "input" / "b.txt").write_text("The fictional access code for this test is Q7F9.", encoding="utf-8")
        run("index", "--root", str(root), "--method", "standard", "--skip-validation")
        marker_in_query(root, "before")
        (root / "input" / "b.txt").unlink()
        run("update", "--root", str(root), "--method", "standard")
        print("UPDATE_DOCUMENT_TABLES", sorted(str(p.relative_to(root)) for p in root.rglob("documents.parquet")), flush=True)
        units = pd.read_parquet(root / "output" / "text_units.parquet")
        print("ORIGINAL_INDEX_MARKER_AFTER_UPDATE", bool(units["text"].str.contains(MARKER, regex=False).any()), flush=True)
        marker_in_query(root, "after-update-original-output")

        clean = Path(tmp) / "clean"
        clean.mkdir()
        setup(clean, base)
        (clean / "input" / "a.txt").write_text("Fictional access instructions: ask the operator for the code.", encoding="utf-8")
        run("index", "--root", str(clean), "--method", "standard", "--skip-validation")
        marker_in_query(clean, "clean-reindex")


if __name__ == "__main__":
    main()
