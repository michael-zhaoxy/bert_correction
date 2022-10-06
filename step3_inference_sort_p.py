# coding: utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

import time
from tqdm import tqdm
from char_sim import CharFuncs
from config import *
from smbert.data.smbert_dataset import DataFactory
from nltk.util import ngrams
import pickle
import re
from smbert.layers.SM_Bert_mlm import SMBertMlm
from transformers import BertConfig

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
        # self.model = torch.load(FinetunePath).to(device).eval()
        config = BertConfig.from_json_file('/Users/xmly/PycharmProjects/smbert_corr/checkpoint/t3/bert_config_L3.json')
        config.device = 'cpu'
        self.model = SMBertMlm(config)
        self.model.device = 'cpu'
        self.model.load_state_dict(torch.load(os.path.join('/Users/xmly/PycharmProjects/smbert_corr/checkpoint/t3/gs203412.pkl'), map_location='cpu'))
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

    def get_topk(self, text):
        input_len = len(text)
        text2id, position, segments = self.get_id_from_text(text)
        with torch.no_grad():
            result = []
            output_tensor = self.model(text2id, position, segments)[:, 1:input_len + 1, :]
            output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
            output_topk_prob = torch.topk(output_tensor, 3).values.squeeze(0).tolist()
            output_topk_indice = torch.topk(output_tensor, 3).indices.squeeze(0).tolist()
            for i, words in enumerate(output_topk_indice):
                tmp = []
                for j, candidate in enumerate(words):
                    word = self.smbert_data.tokenizer.id_to_token(candidate)
                    tmp.append(word)
                result.append(tmp)
        return result, output_topk_prob

    def inference_single(self, text, gt, label=''):
        candidates, probs = self.get_topk(text)
        text_list = list(text)
        result = {
            '原句': text,
            '纠正': [],
            '纠正句子': [],
            '目标': label

        }
        for i, ori in enumerate(text_list):
            if i in gt or (ori == candidates[i][0] and probs[i][0] > 0.999):
                result['纠正'].append([ori])
            elif probs[i][0] > 0.999:
                result['纠正'].append([candidates[i][0]])
            else:
                candidate = candidates[i]
                tmp_candidate = []
                prob = probs[i]
                for j, p in enumerate(prob):
                    if p > 0.0001:
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
        tmp_prop = trigram_model.prob(item)
        if tmp_prop == 0:
            tmp_prop = 1e-7
        prop_q *= tmp_prop
    return prop_q


def threshold(query):
    len_query = len(query)
    if (len_query > 14 or len_query < 2):
        return 1

    badcase = ["郭德刚", "单田方", "将勋", "簌簌清扬", "麦克狐", "狗带猫铃", "迈克胡", "将小牙", "里的爸爸",
               "武练巅峰"]
    if query in badcase:
        return 1
    thresholds = [3.e-06, 1.8e-06, 1.1e-06, 8.3e-07, 3.8e-07, 2.6e-07, 2.0e-07, 1.6e-07, 8.e-08, 8.e-08, 8.e-08, 8.e-08,
                  8.e-08, 8.e-08, 8.e-08]

    return thresholds[len_query - 2]


# def in_gt(str1, str2, gt):
#     if len(gt) == 0:
#         return False
#     diff = []
#
#     for i in range(len(str1)):
#         c1,c2 = str1[i], str2[i]
#         if c1!=c2:
#             diff.append([i])
#
#     for _ in diff:
#         if _ in gt:
#             return True
#     return False


if __name__ == '__main__':
    bert_infer = Inference()
    trigram_model = readbunchobj('/Users/xmly/PycharmProjects/query/correction/slm/trigram.model')
    print('模型加载完毕...')
    # f = open('./data/test_data/recall_test_plus_priori.txt', 'r', encoding='utf-8')
    f = open('/Users/xmly/PycharmProjects/query/intent/data/result_333.txt', 'r', encoding='utf-8')
    # f = open('data/test_data/precision_label.txt', 'r', encoding='utf-8')
    right_cnt = 0
    all_num = 0
    threshold_cnt = 0
    s_time = time.time()

    result_line = []

    ids_model = CharFuncs('char_data/char_meta.txt')

    cache = set()


    for line in (f):
        if line:
            line = line.strip().split('<:>')[0]
            start, end = 0, 0

            all_num += 1
            original_text = line
            if len(line) > 14:
                continue

            gt = set()
            for _ in range(start):
                gt.add(_)
            for _ in range(len(original_text) - end, len(original_text)):
                gt.add(_)


            for _ in range(len(original_text)):
                if _ in gt:
                    continue
                if re.match('[^\u4e00-\u9fa50-9a-zA-Z]+', original_text[_]):
                    gt.add(_)

            list_han = re.findall('[\u4e00-\u9fa5]', original_text)
            if len(list_han) < 2:
                continue

            result = bert_infer.inference_single(original_text, gt)
            if len(result['纠正句子']) == 1 and result['纠正句子'][0] == original_text:
                continue

            label = result['纠正句子'][0]

            # max_prop = 0.0
            # if original_text==label:
            #     max_prop = getScore(original_text)
            # else:
            #     max_prop = getScore(original_text)
            max_prop = getScore(original_text)
            max_res = original_text




            for query in result['纠正句子']:
                if query==max_res:
                    continue
                simi_score, diff_cnt = ids_model.similarity_seqs(line, query)
                if (simi_score < 0.52 or query == max_res):
                    continue
                prop_q = getScore(query)
                if (prop_q > 1e-10 and prop_q > max_prop*pow(10, diff_cnt)) or (prop_q > max_prop*pow(10, diff_cnt*2)):
                    max_prop = prop_q
                    max_res = query

            if max_res != original_text:
                if max_res == label:
                    # max_res = max_res + "-" + "1"
                    tmp_score = float(1.0e-10 / pow(10, (len(max_res)+3)))
                    if max_prop < tmp_score:
                        continue
                else:
                    if max_prop < float(1.0e-10 / pow(10, len(max_res)-1)):
                        continue

                if max_prop < 1e-22:
                    continue


    print('消耗时间：{}'.format((time.time() - s_time) / all_num))

    # with open('data/result/bertPrecision.txt', 'w')as f:
    #     for _ in result_line:
    #         f.write(_)
    #         f.write('\r\n')

    print(len(result_line))
