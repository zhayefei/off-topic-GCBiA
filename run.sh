tar zxvf model.tar.gz

model=model/off-topic-gcbia
tokenizer=model/tokenizer.pkl2

input_file='data/predict.txt'
output_file='output/off-topic.output.txt'

python3.7 predict.py \
    --load-model $model \
    --load-tokenizer $tokenizer \
    --input-file $input_file \
    --output-file $output_file
