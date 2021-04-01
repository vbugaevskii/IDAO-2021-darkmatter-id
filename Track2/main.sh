SCRIPT_DIR=$(cd -- "$(dirname $0)" && pwd)
export PATH=/usr/conda/bin:"${SCRIPT_DIR}":"$PATH"
python main.py --model models
