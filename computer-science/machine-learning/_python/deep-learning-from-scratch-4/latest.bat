@echo off
SET repo_path=deep-learning-from-scratch-4

IF NOT EXIST "%repo_path%" (
    echo Repository not found. Cloning...
    git clone https://github.com/oreilly-japan/deep-learning-from-scratch-4.git
) ELSE (
    echo Repository found. Updating...
    git -C %repo_path% pull
)
