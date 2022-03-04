#!/bin/sh

#
# Copyright (c) 2021 EIDOSLab. All rights reserved.
# This file is part of the EIDOSearch library.
# See the LICENSE file for licensing terms (BSD-style).
#

# Bash script to run multiple wandb sweep commands

helpFunction() {
  echo ""
  echo "Usage: $0 -s sweep"
  echo -e "\t-s wandb sweep command as user/project/sweep_id"
  exit 1 # Exit script after printing help
}

if [ $# -eq 0 ]; then
  helpFunction
fi

while getopts "s:" opt; do
  case $opt in
  s) sweep="$OPTARG" ;;
  ?) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

# GPU idxs
gpus=(2 3 4 5 6 7) # 2 3 4 5 6 7)

# CPU pinning: each gpus has X cpu cores dedicated
cpus=15

docker pull eidos-service.di.unito.it/bragagnolo/zero-grad:latest
j=0

for i in "${gpus[@]}"; do
  start_cpu=$((j * cpus))
  end_cpu=$((start_cpu + cpus - 1))
  echo "GPU ${i} pinning cpus $start_cpu-$end_cpu"
  # .ai https://api.wandb.ai - 36282e74c6260ae7f1c0c91508785e3c93e59fc9
  # service http://eidos-service.di.unito.it:8080 - local-970b63b2a6d40030c6a96ea00f8b540c82947126
  docker run -d --rm -v /data:/data --gpus device=${i} -e WANDB_BASE_URL=https://api.wandb.ai -e WANDB_API_KEY=36282e74c6260ae7f1c0c91508785e3c93e59fc9 --cpuset-cpus $start_cpu-$end_cpu --shm-size 8G --hostname $HOSTNAME --entrypoint /bin/sh eidos-service.di.unito.it/bragagnolo/zero-grad:latest -c "wandb agent ${sweep}"
  j=$((j+1))
done
