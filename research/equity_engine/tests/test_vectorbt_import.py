from importlib.metadata import version


def test_vectorbt_imports_with_supported_plotly_dependency() -> None:
    import vectorbt as vbt

    assert vbt.__version__ == "1.1.0"
    assert version("plotly") == "6.9.0"
