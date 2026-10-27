"""Synthetic installed-package control-flow probe; no model/network calls."""
import asyncio
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pandas as pd
from graphrag.index.update.incremental_index import get_delta_docs
from graphrag.index.workflows import load_update_documents as workflow_module


@dataclass
class Doc:
    title: str
    text: str


class Reader:
    def __init__(self, docs):
        self.docs = docs

    async def __aiter__(self):
        for doc in self.docs:
            yield doc


class Tables:
    def __init__(self, docs):
        self.docs = pd.DataFrame([asdict(doc) for doc in docs])
        self.writes = []

    async def read_dataframe(self, name):
        assert name == "documents"
        return self.docs.copy()

    async def write_dataframe(self, name, value):
        self.writes.append((name, value.copy()))


async def run_case(name, before, after, expected):
    source = Tables(before)
    delta = await get_delta_docs(pd.DataFrame([asdict(x) for x in after]), source)
    actual_new = delta.new_inputs["title"].tolist()
    actual_deleted = delta.deleted_inputs["title"].tolist()
    output = await workflow_module.load_update_documents(Reader(after), source)
    assert (actual_new, actual_deleted, output["title"].tolist()) == expected
    print(f"{name}: computed_new={actual_new}, computed_deleted={actual_deleted}, returned={output['title'].tolist()}")


async def main():
    a, b, c = Doc("a", "alpha"), Doc("b", "fictional marker"), Doc("c", "charlie")
    await run_case("unchanged", [a, b], [a, b], ([], [], []))
    await run_case("delete_only", [a, b], [a], ([], ["b"], []))
    await run_case("delete_plus_add", [a, b], [a, c], (["c"], ["b"], ["c"]))
    await run_case("modify_same_title", [a, b], [a, Doc("b", "changed")], ([], [], []))
    print("4/4 source-function interface checks passed; not full pipeline or query")


if __name__ == "__main__":
    asyncio.run(main())
