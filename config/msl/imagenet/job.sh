#!/bin/bash
#SBATCH --job-name=imagenet-moco
#SBATCH --output=/home-mscluster/dtorpey/code/moco/config/msl/imagenet/log.out
#SBATCH --error=/home-mscluster/dtorpey/code/moco/config/msl/imagenet/log.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=stampede

cd /home-mscluster/dtorpey/code/moco

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m moco.train --config_path /home-mscluster/dtorpey/code/moco/config/msl/imagenet/config.yaml
