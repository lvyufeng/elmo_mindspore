
#!/bin/bash
EXEC_PATH=$(pwd)
test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
    export START_ID=0
}
test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
    export START_ID=4
}

test_dist_${RANK_SIZE}pcs

