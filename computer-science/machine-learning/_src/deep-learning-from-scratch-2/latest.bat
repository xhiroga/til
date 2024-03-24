@echo off
SET repo_path=deep-learning-from-scratch

IF NOT EXIST "%repo_path%" (
    echo Repository not found. Cloning...
    git clone https://github.com/oreilly-japan/deep-learning-from-scratch-2.git
) ELSE (
    echo Repository found. Updating...
    cd %repo_path%
    git pull
)
