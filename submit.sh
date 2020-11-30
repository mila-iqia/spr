#!/bin/bash

export TIME=$1
shift
export GPUS=$1
shift
export JOB_NAME=$1
shift

export DL_ARGS="$@"

echo ${JOB_NAME}
echo ${DL_ARGS}

if [[ $HOSTNAME == cedar* ]]; then
    CPUS=$(expr $GPUS \* 6)
    MEM=$(expr $GPUS \* 24)
    target=cedar_job.sh
else # beluga
    CPUS=$(expr $GPUS \* 10)
    MEM=$(expr $GPUS \* 32)
    target=job.sh
fi
echo ${MEM}
echo ${CPUS}
echo ${GPUS}

sbatch --job-name=${JOB_NAME} -o ${JOB_NAME}.%j.out --export=ALL,DL_ARGS="$DL_ARGS" --account=rrg-bengioy-ad --time=${TIME}:59:00 --ntasks=1 --cpus-per-task=${CPUS} --gres=gpu:${GPUS} --mem=${MEM}G ${target}

echo
