#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='textlabs_agent_gym',
      version='0.0.1',
      packages=find_packages(),
      scripts=[
          "scripts/tl-make-lab-games",
          "scripts/shrink-episode-log"
      ],
      install_requires=open("requirements.txt").readlines())