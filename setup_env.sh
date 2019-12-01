#!/bin/bash

usage()
{
    echo "usage: setup_env.sh
        [-t|--test-env <build test environment - intended for automated testing env>]
        [-e|--env-dir <directory to create environment in. Default:'env'>]
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
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

##### Set defaults if unset
PYTHON=${PYTHON:='python3'}
PYTHON_VERSION=$(${PYTHON} -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')
PYTHON_VERSION_COMPACT="${PYTHON_VERSION//.}"
echo "PYTHON VERSION: ${PYTHON_VERSION}"

if [[ $ENV_DIR ]]
then
    envFolder_str=$ENV_DIR
else
    envFolder_str="env"
fi

# check execute option is set on shell scripts (needed when pycharm copies files)
find . -name "*.sh" | xargs chmod +x

virtualenv -p ${PYTHON} ${envFolder_str}
source ./${envFolder_str}/bin/activate

pip install --progress-bar off -r ./requirements.txt

# Add modules to python path
SITE_PKG_FOLDER=$(python -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())")
echo "../../../.." > ${SITE_PKG_FOLDER}/optionsdetectordata.pth

deactivate

# allow remote interpreter to access remote system variables
rm ./${envFolder_str}/bin/python
echo "#!/bin/bash -l
${ROOT_DIR}/${envFolder_str}/bin/python3 \"\$@\"" > ./${envFolder_str}/bin/python_sys_env_wrapper.sh
chmod 775 ./${envFolder_str}/bin/python_sys_env_wrapper.sh
ln -s python_sys_env_wrapper.sh ./${envFolder_str}/bin/python

printf "\nVirtual environment '$envFolder_str' ready.\nEnter 'source $envFolder_str/bin/activate' to activate.\nEnter 'deactivate' to exit.\n"
