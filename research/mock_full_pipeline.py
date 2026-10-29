"""Offline functional GraphRAG control. Mocks are not evidence of model quality."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def run(*args):
    cmd = [sys.executable, "-m", "graphrag", *args]
    print("RUN", " ".join(args), flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=240)
    if result.returncode:
        print((result.stdout + "\n" + result.stderr)[-9000:], flush=True)
        raise RuntimeError(f"command failed ({result.returncode})")
    print("PASS", args[0], flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="graphrag2590_") as temp:
        root = Path(temp)
        run("init", "--root", str(root))
        config_path = root / "settings.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        extract = '("entity"<|>ALICE<|>PERSON<|>Alice has fictional marker Q7F9)##("entity"<|>TEAM<|>ORGANIZATION<|>A fictional team)##("relationship"<|>ALICE<|>TEAM<|>Alice works with Team<|>5)##<|COMPLETE|>'
        report = json.dumps({"title": "Fictional team", "summary": "Test only", "rating": 1.0,
                             "rating_explanation": "Synthetic", "findings": [{"summary": "Fixture", "explanation": "No real data"}]})
        def mock(response):
            return {"type": "mock", "model_provider": "openai", "model": "gpt-4o-mini", "mock_responses": [response]}
        config["completion_models"] = {"extract": mock(extract), "summary": mock("Fictional description"),
                                       "report": mock(report), "query": mock("Fictional response")}
        config["embedding_models"] = {"emb": {"type": "mock", "model_provider": "openai", "model": "text-embedding-3-small", "mock_responses": [0.1, 0.2, 0.3, 0.4]}}
        config["extract_graph"]["completion_model_id"] = "extract"
        config["extract_graph"]["max_gleanings"] = 0
        config["summarize_descriptions"]["completion_model_id"] = "summary"
        config["community_reports"]["completion_model_id"] = "report"
        config["embed_text"]["embedding_model_id"] = "emb"
        for key in ("local_search", "global_search", "drift_search", "basic_search"):
            config[key]["completion_model_id"] = "query"
            if "embedding_model_id" in config[key]:
                config[key]["embedding_model_id"] = "emb"
        config["vector_store"]["index_schema"] = {name: {"vector_size": 4} for name in ("text_unit_text", "entity_description", "community_full_content")}
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        inputs = root / "input"
        inputs.mkdir(exist_ok=True)
        (inputs / "a.txt").write_text("Alice and Team are fictional coworkers.", encoding="utf-8")
        (inputs / "b.txt").write_text("Alice has fictional marker Q7F9.", encoding="utf-8")
        run("index", "--root", str(root), "--method", "standard", "--skip-validation")
        before = pd.read_parquet(root / "output" / "documents.parquet")
        assert set(before["title"]) == {"a.txt", "b.txt"}
        (inputs / "b.txt").unlink()
        run("update", "--root", str(root), "--method", "standard")
        after = pd.read_parquet(root / "output" / "documents.parquet")
        print("INDEX_TITLES_BEFORE", sorted(before["title"].tolist()))
        print("INDEX_TITLES_AFTER_DELETION_UPDATE", sorted(after["title"].tolist()))
        assert "b.txt" in set(after["title"]), "unexpected change: re-evaluate issue"
        print("OBSERVED_STALE_DOCUMENT_AFTER_UPDATE mock-mode; actual model query not tested")


if __name__ == "__main__":
    main()
