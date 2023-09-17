#!/usr/bin/env bash
set -euxo pipefail
module load cuda/12.1
source /home/mcharusaie/myvenv/bin/activate
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 CIFAR10K.py
