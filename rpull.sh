#!/bin/bash
rsync -av --progress Stanage:/mnt/parscratch/users/smp24rme/ezcascades/ . --exclude='.git/*' --exclude='*.ipynb_checkpoints'