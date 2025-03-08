import json
# 返回的字符串包含有关异常的详细信
import traceback
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
import sys
sys.path.append('ptune_chatglm/glm_config.py')

from glm_config import *
#-下面是bert的输入格式转换
# def convert_example(
#         examples: dict,
#         tokenizer,
#         max_source_seq_len: int,
#         max_target_seq_len: int,
# ):
#     """
#     将样本数据转换为适用于BERT模型的输入数据。
#
#     Args:
#         examples (dict): 训练数据样本
#         max_source_seq_len (int): 最大长度限制
#         max_target_seq_len (int): 最大目标长度限制
#
#     Returns:
#         dict -> tokenized_output = {
#                             'input_ids': [[1525, 10, ...], [758, 2345, ...]],
#                             'attention_mask': [[1, 1, ...], [1, 1, ...]],
#                             'token_type_ids': [[0, 0, ...], [0, 0, ...]],
#                             'labels': [[822, 10, ...], [125, 58...]]
#                         }
#     """
#     tokenized_output = {
#         'input_ids': [],
#         'attention_mask': [],
#         'token_type_ids': [],
#         'labels': []
#     }
#
#     max_seq_length = 300  # 设置最大序列长度为300，包含 [CLS] 和 [SEP] 两个
#     total_special_tokens = 3  # [CLS] + [SEP] + [SEP]
#
#     # 为了确保长度不超过300，我们需要将 max_source_seq_len 和 max_target_seq_len 调整为 297
#     max_source_seq_len = min(max_source_seq_len, max_seq_length - total_special_tokens)
#     max_target_seq_len = min(max_target_seq_len, max_seq_length - total_special_tokens)
#
#     for example in examples['text']:
#         try:
#             example = json.loads(example)
#             context = example["context"]
#             target = example["target"]
#
#             context_ids = tokenizer.encode(
#                 text=context,
#                 add_special_tokens=False
#             )
#             target_ids = tokenizer.encode(
#                 text=target,
#                 add_special_tokens=False
#             )
#
#             # 截断以适应最大长度
#             if len(context_ids) >= max_source_seq_len:
#                 context_ids = context_ids[:max_source_seq_len]
#             if len(target_ids) >= max_target_seq_len:
#                 target_ids = target_ids[:max_target_seq_len]
#
#             input_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + target_ids + [tokenizer.sep_token_id]
#
#             # 确保 input_ids 长度为 300
#             pad_len = max_seq_length - len(input_ids)
#             if pad_len < 0:  # 如果超过300，直接截断
#                 input_ids = input_ids[:max_seq_length]
#             else:  # 填充至300
#                 input_ids += [tokenizer.pad_token_id] * pad_len
#
#             # 构建 attention_mask: [1, 1, ..., 1]，长度与 input_ids 一致
#             attention_mask = [1] * len(input_ids)
#
#             # token_type_ids: [0, 0, ..., 0] (用于区分 context 和 target)，context 部分为 0，target 部分为 1
#             token_type_ids = [0] * (len(context_ids) + 1) + [1] * (len(target_ids) + 1)
#
#             # labels: 目标部分的 ID，context 部分用 -100 填充
#             labels = [-100] * (len(context_ids) + 1) + target_ids + [-100]
#
#             tokenized_output['input_ids'].append(input_ids)
#             tokenized_output['attention_mask'].append(attention_mask)
#             tokenized_output['token_type_ids'].append(token_type_ids)
#             tokenized_output['labels'].append(labels)
#
#         except Exception as e:
#             print(f"Error processing example: {e}")
#             continue
#
#     for k, v in tokenized_output.items():
#         tokenized_output[k] = np.array(v)
#
#     return tokenized_output


# #chatglm的输入是partA+PartB的格式，因此要把context和target合并，同时合并时有一定规则
def convert_example(
        examples: dict,
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将样本数据转换为Prompt-tuning模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "target": "2017年银行贷款基准利率"}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): prompt最大长度   提示文本的最大长度
        max_target_seq_len (int): 答案最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }

    max_seq_length = max_source_seq_len + max_target_seq_len  #原始样本和目标样本相加就是最大句子长度

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]
            # print(f'context-->\n{context}')  #context就是训练集中instruction和input
            # print(f'target-->\n{target}')   #target就是训练集中answer，做信息抽取
            # break
            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )
            # break
            print(f'prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')

            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )
            print(f'target_ids--》{target_ids}\n{len(target_ids)}')
            # print('37010-->', tokenizer.convert_ids_to_tokens([37010, 12,5, 76331, 83362]))
            #对prompt输入如果超过设置的最大长度截断
            #这里多留一位置给gmask_token_id：130001    用gmask因为这里模型相当于预测的是一个句子，如果预测的是一个词 就用"mask_token_id": 130000,
            if len(prompts_ids) >= max_source_seq_len:                                          # source 需要留一个 [gMASK] token 在结尾
                prompts_ids = prompts_ids[:max_source_seq_len - 1]
            #这里多留两个位置给"bos_token_id": 130004,和"eos_token_id": 130005,
            if len(target_ids) >= max_target_seq_len - 1:                                       # target 需要留一个 <sop> 在开头和一个 <eop> token 在结尾
                target_ids = target_ids[:max_target_seq_len - 2]
            # print(f'new_prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')
            # print(f'new_target_ids--》{target_ids}\n{len(target_ids)}')
            # a = tokenizer.convert_tokens_to_string(target_ids)
            # print(a)

            #cahtglm的输入是要特殊字符的
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)     # source_ids + [gMASK] + <sop>[也是bos] + target_ids + <eop>
            # print(f'input_ids-->{input_ids}')
            # print(f'input_ids-->{len(input_ids)}')


            context_length = input_ids.index(tokenizer.bos_token_id)                            # bos 在 target 的第一位
            # print(f'context_length-->{context_length}')


            mask_position = context_length - 1                                                  # [gMASK] 在 source 的最后一位
            labels = [-100] * context_length + input_ids[mask_position + 1:]                    # 从 bos 开始到后面所有的 target 到 eos 都为 label

            pad_len = max_seq_length - len(input_ids)
            # print(f'pad_len-->{pad_len}')

            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            print(f'input_ids-->{input_ids}\n{len(input_ids)}')
            labels = labels + [-100] * pad_len
            print(f'labels-->{labels}\n{len(labels)}')


            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output

def get_max_length(
        tokenizer,
        dataset_file: str
    ):
    """
    测试数据集最大的输入/输出tokens是多少。

    Args:
        dataset_file (str): _description_
    """
    source_seq_len_list = []
    target_seq_len_list = []
    with open(dataset_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")

if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})
    print(type(train_dataset))
    print(train_dataset)
    # print('*'*80)
    # print(train_dataset['train'])
    # print('*'*80)
    # print(train_dataset['train']['text'])
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    tokenized_output = convert_example(examples=train_dataset['train'],
                                       tokenizer=tokenizer,
                                       max_source_seq_len=30,
                                       max_target_seq_len=20)
    # print(len(tokenized_output["input_ids"][0]))
    # print(len(tokenized_output["labels"][0]))

    get_max_length(tokenizer, pc.train_path)