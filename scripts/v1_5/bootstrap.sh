export PYTHONPATH=$PWD:$PYTHONPATH
set -e

mode=${1}
args="${@:2}"

if [[ ${mode} == 'train' ]]; then
    scripts/v1_5/TORCHRUN ${args}
else
    python3 $args 
fi
