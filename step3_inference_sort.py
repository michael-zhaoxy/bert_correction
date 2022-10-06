# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

import time
from tqdm import tqdm
from char_sim import CharFuncs
from config import FinetunePath, device, PronunciationPath, SentenceLength
from smbert.data.smbert_dataset import DataFactory
from nltk.util import ngrams
import pickle

def curve(confidence, similarity):
    flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
    flag2 = 0.1 * confidence + similarity - 0.6 > 0
    if flag1 or flag2:
        return True
    return False


class Inference(object):
    def __init__(self, mode='s'):
        self.char_count = 0
        self.top1_acc = 0
        self.top5_acc = 0
        self.sen_count = 0
        self.sen_acc = 0
        self.sen_invalid = 0
        self.sen_wrong = 0
        self.mode = mode
        self.model = torch.load(FinetunePath).to(device).eval()
        self.model.device = device
        self.smbert_data = DataFactory()
        print('加载模型完成！')

    def get_id_from_text(self, text):
        assert isinstance(text, str)
        ids = self.smbert_data.tokenizer.tokens_to_ids(text)
        inputs = [self.smbert_data.token_cls_id] + ids + [self.smbert_data.token_sep_id]
        position = [i for i in range(len(inputs))]
        segments = [1 for x in inputs]

        inputs = torch.tensor(inputs).unsqueeze(0).to(device)
        position = torch.tensor(position).unsqueeze(0).to(device)
        segments = torch.tensor(segments).unsqueeze(0).to(device)
        return inputs, position, segments

    # 末尾的[SEP]是否需要，做实验
    def get_topk(self, text):
        input_len = len(text)
        text2id, position, segments = self.get_id_from_text(text)
        with torch.no_grad():
            result = []
            output_tensor = self.model(text2id, position, segments)[:, 1:input_len + 1, :]
            output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
            output_topk_prob = torch.topk(output_tensor, 5).values.squeeze(0).tolist()
            output_topk_indice = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
            for i, words in enumerate(output_topk_indice):
                tmp = []
                for j, candidate in enumerate(words):
                    word = self.smbert_data.tokenizer.id_to_token(candidate)
                    tmp.append(word)
                result.append(tmp)
        return result, output_topk_prob

    def inference_single(self, text, gt=''):
        candidates, probs = self.get_topk(text)
        text_list = list(text)
        result = {
            '原句': text,
            '纠正': [],
            '纠正句子': [],
            '目标': gt

        }
        for i, ori in enumerate(text_list):
            if ori == candidates[i][0] and probs[i][0] > 0.999:
                result['纠正'].append([ori])
            else:
                candidate = candidates[i]
                tmp_candidate = []
                prob = probs[i]
                for j, p in enumerate(prob):
                    if p > 0.001:
                        tmp_candidate.append(candidate[j])
                result['纠正'].append(tmp_candidate)

        for i in range(len(text_list)):
            line = result['纠正句子']
            if i == 0:
                line.append('')
            tmp_line = result['纠正'][i]
            result['纠正句子'] = []
            for tmp in tmp_line:
                for _ in line:
                    str = _ + tmp
                    result['纠正句子'].append(str)

        return result

    def inference_batch(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        res = []
        for line in tqdm(f):
            if line:
                line = line.strip()
                self.sen_count += 1
                line = line.split('<:>')
                src = line[1]
                target = line[0]
                result = self.inference_single(target, src)
                list1 = []
                list1.append(result['原句'])
                list1.append(result['目标'])
                for _ in result['纠正句子']:
                    list1.append(_)
                res.append('<:>'.join(list1))
        return res

def readbunchobj(path):
    file_obj = open(path, 'rb')
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch

def getScore(query):
    trigrams_q = ngrams(list(query), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                        right_pad_symbol='<e>')
    prop_q = 1
    for item in trigrams_q:
        prop_q *= trigram_model.prob(item)
    return prop_q

if __name__ == '__main__':
    bert_infer = Inference()
    trigram_model = readbunchobj('/Users/xmly/PycharmProjects/query/correction/slm/trigram.model')
    print('模型加载完毕...')
    # f = open('./data/test_data/recall_test_plus.txt', 'r', encoding='utf-8')
    f = open('./data/test_data/precision_label.txt', 'r', encoding='utf-8')
    right_cnt = 0
    all_num = 0
    s_time = time.time()
    ids_model = CharFuncs('char_data/char_meta.txt')
    result_line = []

    for line in (f):
        if line:
            line = line.strip()
            line = line.split('<:>')
            src = line[0]
            label = line[1]
            if len(src) > 14:
                continue
            all_num += 1
            result = bert_infer.inference_single(src, '')

            max_prop = getScore(src)
            max_res = src

            for _ in result['纠正句子']:

                if(ids_model.similarity_seqs(_, src)[0] < 0.52 or _ == src):
                    continue

                prop_q = getScore(_)
                if prop_q > max_prop:
                    max_prop = prop_q
                    max_res = _
            result_line.append('{}<:>{}<:>{}'.format(src, label, max_res))
            # result_line.append('{}<:>{}'.format(src, max_res))
            if (max_res == label):
                right_cnt += 1
            else:
                print('{}\t{}\t{}'.format(src, label, max_res))

    print(right_cnt/all_num)
    print('平均消耗时间：{}'.format((time.time()-s_time)/all_num))

    # with open('data/result/bertRecall_plus.txt', 'w')as f:
    #     for _ in result_line:
    #         f.write(_)
    #         f.write('\r\n')



