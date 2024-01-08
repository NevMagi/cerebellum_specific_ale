#!/bin/bash
# Prints out the .submit instructions for HTCondor jobs given any bash script
# Usage: ./gen_submit_bash.sh <bash_script_full_path> <bash_script_args> | condor_submit

PROJECT_DIR="/data/project/cerebellum_ale"

bash_path=$1
shift
LOGS_DIR=$1
shift
args=$*
if [[ "$LOGS_DIR" == "default" ]]; then
    LOGS_DIR="${PROJECT_DIR}/logs/misc_bash"
fi
RAM='16G'

# create the logs dir if it doesn't exist
[ ! -d "${LOGS_DIR}" ] && mkdir -p "${LOGS_DIR}"

# print the .submit header
printf "# The environment
universe       = vanilla
getenv         = True
request_gpus   = 1
request_memory = ${RAM}

# Execution
initial_dir    = ${PROJECT_DIR}
executable     = /bin/bash
\n"

shift 1
printf "arguments = ${bash_path} ${args} \n"
printf "log       = ${LOGS_DIR}/\$(Cluster).\$(Process).log\n"
printf "output    = ${LOGS_DIR}/\$(Cluster).\$(Process).out\n"
printf "error     = ${LOGS_DIR}/\$(Cluster).\$(Process).err\n"
printf "Queue\n\n"