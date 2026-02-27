import os


def test_project_structure():
    assert os.path.isdir("ingestion")
    assert os.path.isdir("transformations")
    assert os.path.isdir("orchestration")