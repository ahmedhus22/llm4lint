import pytest
from unittest.mock import (
    mock_open,
    patch,
    call
)
from llm4lint.llm4lint import App

@pytest.fixture
def app_mock(capsys, monkeypatch):
    app = App("llm4lint7b")
    sample_code = """l=[1,2,3]
for i in l:
    l.append(i)
"""
    m = mock_open()
    with patch("builtins.open", m):
        yield {
            "app": app,
            "sample_code": sample_code,
            "mock_file": m,
            "capsys": capsys,
            "monkeypatch": monkeypatch
        }


def test_code_lines(app_mock):
    sample_code = """def test():
    msg = 'Testing...'
    print(msg)"""
    sample_code_lines = """1   def test():
2       msg = 'Testing...'
3       print(msg)
"""
    assert sample_code_lines == app_mock["app"]._addcodelines(sample_code)

def test_log_code(app_mock):
    msg = """# Test for parsing and logging code blocks
first block
```python
print('block1')
```
second block
```
print('block2')
```
"""
    app_mock["app"]._logcode(msg)
    expected_calls = [
        call("print('block1')"),
        call("print('block2')")
    ]
    app_mock["mock_file"]().write.assert_has_calls(expected_calls)

@pytest.mark.slow
def test_init_shell(app_mock, mocker):
    """only launches, prompts and closes the shell"""
    mock_get_code = mocker.patch.object(App, "_getcode", return_value=app_mock["sample_code"])
    prompts = iter(["suggest improvements", "q"])
    app_mock["monkeypatch"].setattr("builtins.input", lambda _: next(prompts))
    app_mock["app"].init_shell(file=None)
    _ = app_mock["capsys"].readouterr()