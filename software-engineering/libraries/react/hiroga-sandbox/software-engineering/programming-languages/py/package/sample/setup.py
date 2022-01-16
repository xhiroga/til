# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='module',
    version='0.0.1',
    description='テスト用に作成したモジュール',
    install_requires=[],
    dependency_links=[]
)