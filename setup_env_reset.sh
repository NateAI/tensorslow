#!/bin/bash

usage()
{
    echo "usage: setup_env.sh
        [-e|--env-dir <directory to create environment in. Default:'env' if -s not passed in >]
        [-p|--python installation to use as base <Default: 'python3'>
        [-h|--help]"
}


##### Main
while [[ "$1" != "" ]]; do
    case $1 in
        -e | --env-dir )        shift
                                ENV_DIR=$1
                                ;;
        -p | --python )         shift
                                PYTHON=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

##### cd to current script directory as we expect the build and run scripts here too
cd "${0%/*}"  # https://stackoverflow.com/questions/3349105/how-to-set-current-working-directory-to-the-directory-of-the-script


##### Set defaults if unset
PYTHON=${PYTHON:='python3'}
if [[ $ENV_DIR ]]
then
    envFolder_str=$ENV_DIR
else
    envFolder_str="env"
fi

# check execute option is set on shell scripts (needed when pycharm copies files)
find . -name "*.sh" | xargs chmod +x

echo "Removing virtual environment ${envFolder_str} ."
rm -rf ./${envFolder_str}


echo "Removing external dependencies (./extern) ."
rm -rf ./extern

echo "Setting up new virtual environment."
./setup_env.sh -e ${envFolder_str} -p ${PYTHON}
