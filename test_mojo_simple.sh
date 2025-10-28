#!/bin/bash
cd /Users/wscholl/vectro
export MODULAR_HOME=/Users/wscholl/vectro/.pixi/envs/default
export PATH=/Users/wscholl/vectro/.pixi/envs/default/bin:$PATH
mojo run src/simple_test.mojo
