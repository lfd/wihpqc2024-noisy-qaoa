QLM_RESOURCEMANAGER_CONFIG_FILE_PATH=/invalid/path python3 run_benchmarks.py \
    -v -r -o "results/bounds" -P $1 --done_marker \
    -p "{metric:'approx_ratio',source:{type:['random','approximation']},\
    graphs:{problem:['maxcut', 'partition'],size:[5,6,7,8,9,10]}}" 