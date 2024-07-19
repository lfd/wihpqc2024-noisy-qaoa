QLM_RESOURCEMANAGER_CONFIG_FILE_PATH=/invalid/path python3 run_benchmarks.py \
    -v -r -o "results/main_evaluation" -P $1 --done_marker -c \
    -p "{metric:'approx_ratio',source:{model:[\
        {type:'ideal'},\
        {type:'noisy_composite', noise: [0.25, 0.5, 0.75, 1.0], time: 0, sx: true},\
        {type:'noisy_composite', time: [0.25, 0.5, 0.75, 1.0], noise: 0, sx: true},\
        {type:'noisy_composite', noise: [0.25, 0.5, 0.75, 1.0], time: [0.25, 0.5, 0.75, 1.0], sx: true}\
    ], algorithm:[{type:['qaoa','wsqaoa','wsinitqaoa'],nLayers:[1,2,3,4], nShots:3}, {type:'rqaoa', nSamples:10, nShots: 5, nLayers:[1,2,3,4]}]},\
    graphs:{problem:['maxcut', 'partition'],size:[5,6,7,8,9,10]}}" 