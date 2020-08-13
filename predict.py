#! encoding=utf-8
import sys
import pickle as pkl
import logging as logger
from numpy import squeeze
import argparse

from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from util.data_processing import table_tokenizer
import config

def load_model(model_weights):
    model = tf.saved_model.load(model_weights)
    logger.info('Loaded model succ')
    return model

def load_model_and_tokenizer(model_weights, tokenizer_pkl):
    """
    load model & tokenizer
    """
    with open(tokenizer_pkl, 'rb') as fin:
        tokenizer = pkl.load(fin)
    vocab_size = len(tokenizer.word_index) + 1
    model = load_model(model_weights)
    return model, tokenizer

def _make_text2seq(texts, text_len, tokenizer):
    texts = table_tokenizer(texts, uncase=True)
    text_seq = tokenizer.texts_to_sequences(texts)
    text_seq = pad_sequences(text_seq,
                            maxlen=text_len,
                            truncating=u'post'
                            )
    text_seq = tf.convert_to_tensor(text_seq, dtype=tf.int32)
    return text_seq

def predict(prompt_list, response_list, tokenizer, model):
    pmpt_seq = _make_text2seq(prompt_list, config.prompt_len, tokenizer)
    rspn_seq = _make_text2seq(response_list, config.response_len, tokenizer)
    input_seq = [pmpt_seq, rspn_seq]
    model = model.signatures["serving_default"]
    preds = model(input_1=pmpt_seq, input_2=rspn_seq)['dense_4']
    preds = squeeze(preds)
    logger.info("preds:{}".format(preds))
    return preds

def predict_interactive(model_weights, tokenizer_pkl):
    """
    用户输入任意prompt & response, 模型判断是否离题
    """
    logger.info("Welcome to general off-topic.")
    model, tokenizer  = load_model_and_tokenizer(model_weights, tokenizer_pkl)
    prompt = input("Please input one question:")
    while True:
        response = input("Please input one answer:")
        if type(response) == list and response[0].lower().startswith('quit'):
            break
        result = predict(prompt, response, tokenizer, model)
    return 

def predict_from_file(test_file, model_weights, tokenizer_pkl, output_file):
    prompts, responses = [], []
    model, tokenizer = load_model_and_tokenizer(model_weights, tokenizer_pkl)
    with open(test_file, 'r') as fin:
        for idx, line in enumerate(fin):
            if idx == 0 and line.strip().startswith('prompt'):
                continue
            prompt, response = line.strip().split('\t')
            prompts.append(prompt)
            responses.append(response)
    preds = predict(prompts, responses, tokenizer, model)
    assert len(preds) == len(prompts)
    with open(output_file, 'w') as fout:
        for idx, pred in enumerate(preds):
            cate = '0' if pred < config.THLD else '1'
            write_line = '\t'.join([prompts[idx], responses[idx], str(pred), cate])
            fout.write(write_line + '\n')
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model",
                        dest="load_model", type=str, metavar='<str>',
                        default=None, help="Path to the existing model",
                        required=True)
    parser.add_argument("--load-tokenizer",
                        dest="load_tokenizer", type=str, metavar='<str>',
                        default=None, help="Path to the existing tokenizer",
                        required=True)
    parser.add_argument("--input-file",
                        dest="input_file", type=str, metavar='<str>',
                        default='data/predict.txt', 
                        help="prompt\tresponse format predict file",
                        )
    parser.add_argument("--output-file",
                        dest="output_file", type=str, metavar='<str>',
                        default='output/off-topic.output.txt', help="predict output file",
                        )
    args = parser.parse_args()

    predict_from_file(args.input_file, args.load_model,
                        args.load_tokenizer, args.output_file)
