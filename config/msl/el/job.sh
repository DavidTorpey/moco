#!/bin/bash
#SBATCH --job-name=el-moco
#SBATCH --output=/home-mscluster/dtorpey/code/moco/config/msl/el/log.out
#SBATCH --error=/home-mscluster/dtorpey/code/moco/config/msl/el/log.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=stampede

cd /home-mscluster/dtorpey/code/moco

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m moco.train --config_path /home-mscluster/dtorpey/code/moco/config/msl/el/config.yaml
