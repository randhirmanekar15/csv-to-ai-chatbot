"""Tests for serialize_rows (no embeddings / FAISS required)."""

import pandas as pd

from chatbot import serialize_rows


def test_serialize_rows_pipe_delimits():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    docs = serialize_rows(df)
    assert docs == ["1 | x", "2 | y"]


def test_serialize_rows_count_matches():
    df = pd.DataFrame({"a": range(5)})
    assert len(serialize_rows(df)) == 5
