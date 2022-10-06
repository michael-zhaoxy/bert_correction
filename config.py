import time
import torch
import argparse

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

VocabPath = 'checkpoint/saved/vocab.txt'

# ## mlm模型文件路径 ## #
SourcePath = 'data/test/all_data.txt'
CorpusPath = 'data/test/train.txt'
TestPath = 'data/test/test.txt'
PronunciationPath = 'char_data/char_meta.txt'

# Debug开关
Debug = False

# attention_mask开关
AttentionMask = True

# 使用预训练模型开关
UsePretrain = True

# mask方式
# True表示mask时会mask每个字，而且对于出现频率较低的字会多次mask，频率的限制由WordGenTimes决定
# False表示按照bert的mask方式进行mask
AllMask = False

# ## MLM训练调试参数开始 ## #
MLMEpochs = 16
WordGenTimes = 3
if WordGenTimes > 1:
    RanWrongDivisor = 1.0
else:
    RanWrongDivisor = 0.15
MLMLearningRate = 1e-4
BatchSize = 16
SentenceLength = 16
# FinetunePath = 'checkpoint/saved/mlm_trained_len_{}.model'.format(SentenceLength)
FinetunePath = '/Users/xmly/PycharmProjects/smbert_corr_ori/checkpoint/pretrain_model/mlm_trained_alldata_len_16_4.model'
# ## MLM训练调试参数结束 ## #

# ## MLM通用参数 ## #
DropOut = 0.1
MaskRate = 0.15
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
#HiddenSize = 384
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12
bert_config_file = 'checkpoint/pretrain/bert_config.json'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# def parse(opt=None):
#     parser = argparse.ArgumentParser()
#
#     ## Required parameters
#
#     parser.add_argument("--vocab_file", default=VocabPath, type=str, required=False,
#                         help="The vocabulary file that the BERT model was trained on.")
#     parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str, required=False,
#                         help="The output directory where the model checkpoints will be written.")
#
#     ## Other parameters
#     parser.add_argument("--train_file", default=csc_train_file, type=str, help="SQuAD json for training. E.g., train-v2.0.json")
#     parser.add_argument("--predict_file", default=csc_dev_file, type=str,
#                         help="SQuAD json for predictions. E.g., dev-v2.0.json or test-v2.0.json")
#     parser.add_argument("--do_lower_case", action='store_true',
#                         help="Whether to lower case the input text. Should be True for uncased "
#                              "models and False for cased models.")
#     parser.add_argument("--max_seq_length", default=length, type=int,
#                         help="The maximum total input sequence length after WordPiece tokenization. Sequences "
#                              "longer than this will be truncated, and sequences shorter than this will be padded.")
#     # parser.add_argument("--doc_stride", default=doc_stride, type=int,
#     #                     help="When splitting up a long document into chunks, how much stride to take between chunks.")
#     parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
#     parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
#     parser.add_argument('--do_test', default=False, action='store_true', help="Whether to run eval on the test set.")
#
#     parser.add_argument("--train_batch_size", default=BatchSize, type=int, help="Total batch size for training.")
#     parser.add_argument("--predict_batch_size", default=256, type=int, help="Total batch size for predictions.")
#     parser.add_argument("--learning_rate", default=learning_rate, type=float, help="The initial learning rate for Adam.")
#     parser.add_argument("--num_train_epochs", default=ep, type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--warmup_proportion", default=0.1, type=float,
#                         help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
#                              "of training.")
#     parser.add_argument("--n_best_size", default=20, type=int,
#                         help="The total number of n-best predictions to generate in the nbest_predictions.json "
#                              "output file.")
#     parser.add_argument("--no_cuda",
#                         default=False,
#                         action='store_true',
#                         help="Whether not to use CUDA when available")
#     parser.add_argument('--gradient_accumulation_steps',
#                         type=int,
#                         default=accu,
#                         help="Number of updates steps to accumualte before performing a backward/update pass.")
#     parser.add_argument("--local_rank",
#                         type=int,
#                         default=-1,
#                         help="local_rank for distributed training on gpus")
#     parser.add_argument('--fp16',
#                         default=False,
#                         action='store_true',
#                         help="Whether to use 16-bit float precisoin instead of 32-bit")
#
#     parser.add_argument('--random_seed',type=int,default=torch_seed)
#     # parser.add_argument('--fake_file_1',type=str,default=DA_file)
#     # parser.add_argument('--fake_file_2',type=str,default=None)
#     parser.add_argument('--load_model_type',type=str,default='bert',choices=['bert','all','none'])
#     parser.add_argument('--weight_decay_rate',type=float,default=0.01)
#     parser.add_argument('--do_eval',action='store_true')
#     parser.add_argument('--PRINT_EVERY',type=int,default=200)
#     parser.add_argument('--weight',type=float,default=1.0)
#     parser.add_argument('--ckpt_frequency',type=int,default=ckpt_frequency)
#     parser.add_argument('--tuned_checkpoint_T',type=str,default=trained_teacher_model)
#     parser.add_argument('--tuned_checkpoint_S',type=str,default=None)
#     parser.add_argument("--init_checkpoint_S", default=trained_teacher_model, type=str)
#     parser.add_argument("--bert_config_file_T", default=bert_config_file_T, type=str, required=False)
#     parser.add_argument("--bert_config_file_S", default=bert_config_file_S, type=str, required=False)
#     parser.add_argument("--temperature", default=temperature, type=float, required=False)
#     parser.add_argument("--teacher_cached",action='store_true')
#
#     parser.add_argument('--schedule',type=str,default='slanted_triangular')
#     parser.add_argument('--null_score_diff_threshold',type=float,default=99.0)
#
#     parser.add_argument('--tag',type=str,default='RB')
#     parser.add_argument('--no_inputs_mask',action='store_true')
#     parser.add_argument('--no_logits', action='store_true')
#     parser.add_argument('--output_att_score',default='true',choices=['true','false'])
#     parser.add_argument('--output_att_sum', default='false',choices=['true','false'])
#     parser.add_argument('--output_encoded_layers'  ,default='true',choices=['true','false'])
#     parser.add_argument('--output_attention_layers',default='true',choices=['true','false'])
#     parser.add_argument('--matches',default=['L3_hidden_mse', 'L3_hidden_smmd'] , nargs='*',type=str)
#     global args
#     if opt is None:
#         args = parser.parse_args([])
#     else:
#         args = parser.parse_args(opt)