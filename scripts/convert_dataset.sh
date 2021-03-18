CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/tests/fixtures
# train dataset
python ${CUR_DIR}/elmo/data/reader.py  \
    --vocab_path="${DATA_PATH}/train/vocab.txt" \
    --options_path="${DATA_PATH}/model/options.json"\
    --input_file="${DATA_PATH}/train/data.txt" \
    --output_file="${DATA_PATH}/train.mindrecord"