import random

from common import check_srcdata_and_vocab
from config import *
from smbert.common.tokenizers import Tokenizer


def random_wrong(text):
    tokenizer = Tokenizer(VocabPath)
    length = len(text)
    position = random.randint(0, length - 1)
    number = random.randint(672, 7992)
    text = list(text)
    text[position] = tokenizer.id_to_token(number)
    text = ''.join(text)
    return text


def gen_train_test():
    list = []
    with open(SourcePath) as f:
        line = f.readline()
        while line:
            line_arr = line.strip().split('<:>')
            if len(line_arr) != 2:
                line = f.readline()
                continue
            if len(line_arr[0]) != len(line_arr[1]) or len(line_arr[0]) > 14:
                line = f.readline()
                continue
            list.append(line.strip())
            line = f.readline()

    random.shuffle(list)
    f_train = open(CorpusPath, 'w', encoding='utf-8')
    f_test = open(TestPath, 'w', encoding='utf-8')

    for line in list:
        line = line.strip()
        rad = random.randint(0, 10)
        if rad < 1:
            f_test.write(line + '\n')
        else:
            f_train.write(line + '\n')

    f_train.close()
    f_test.close()


if __name__ == '__main__':
    print(len(open(VocabPath1, 'r', encoding='utf-8').readlines()))
    check_srcdata_and_vocab(SourcePath)
    gen_train_test()
