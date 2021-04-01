SCRIPT_DIR=$(cd -- "$(dirname $0)" && pwd)
export PATH=/usr/conda/bin:"${SCRIPT_DIR}":"$PATH"

date
echo "=========================================="

echo $PATH
echo "=========================================="

pwd
echo "=========================================="

ls -l ${SCRIPT_DIR}
echo "=========================================="

python "${SCRIPT_DIR}/check_imports.py"
echo "=========================================="

pip freeze
echo "=========================================="

python "${SCRIPT_DIR}/utils.py" "${SCRIPT_DIR}/tests"
