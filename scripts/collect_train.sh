#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python ${DIR}/../kitti_parser.py --start-index 0 --num-data 5 --dataset kitti --training
