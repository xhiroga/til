# Reproduction

```console
$ uv run main.py

Traceback (most recent call last):
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/main.py", line 1, in <module>
    import gradio as gr
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/.venv/lib/python3.11/site-packages/gradio/__init__.py", line 3, in <module>
    import gradio._simple_templates
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/.venv/lib/python3.11/site-packages/gradio/_simple_templates/__init__.py", line 1, in <module>
    from .simpledropdown import SimpleDropdown
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/.venv/lib/python3.11/site-packages/gradio/_simple_templates/simpledropdown.py", line 7, in <module>
    from gradio.components.base import Component, FormComponent
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/.venv/lib/python3.11/site-packages/gradio/components/__init__.py", line 22, in <module>
    from gradio.components.datetime import DateTime
  File "/home/hiroga/Documents/GitHub/til/software-engineering/python/gradio/_src/pytz-missing/.venv/lib/python3.11/site-packages/gradio/components/datetime.py", line 9, in <module>
    import pytz
ModuleNotFoundError: No module named 'pytz'
```
