# -*- coding:utf-8 -*-
import torch

class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备：{self.device}")
        # self.device = 'cpu'
        #模型路径
        # self.pre_model = 'D:\BaiduSyncdisk\LLM\ChatGLM-6B'
        self.pre_model = "/root/autodl-tmp/ChatGLM-6B"
        #训练集路径
        # self.train_path = 'D:\BaiduSyncdisk\LLM\ptune_chatglm\data\mixed_train_dataset.jsonl'
        self.train_path = "/root/LoRA_chatglm/data/mixed_train_dataset.jsonl"
        #验证集路径
        # self.dev_path = 'D:\BaiduSyncdisk\LLM\ptune_chatglm\data\mixed_dev_dataset.jsonl'
        self.dev_path = "/root/LoRA_chatglm/data/mixed_dev_dataset.jsonl"
        #是否使用lora
        self.use_lora = True
        #是否使用p-tuing
        self.use_ptuning = False
        #lora低秩矩阵的秩为8，lora是先降维再升维，先降到8维
        self.lora_rank = 8
        #一个批次多少样本
        self.batch_size = 2 #在ChatGLM模型中原本是1
        self.epochs = 2
        self.learning_rate = 3e-5
        #权重衰减
        self.weight_decay = 0
        #学习率预热系数比例
        self.warmup_ratio = 0.06
        #context文本输入限制
        self.max_source_seq_len = 400
        #target文本限制
        self.max_target_seq_len = 300
        # 每隔多少步打印日志
        self.logging_steps = 10
        #每隔多少步保存
        self.save_freq = 600
        #如果使用了p-tuing，就要定义伪tokens的长度，限制模型自己找合适长度的模版
        self.pre_seq_len = 128
        self.prefix_projection = False # 默认为False,即p-tuning,如果为True，即p-tuning-v2
        #模型保存路径
        self.save_dir = '/root/autodl-tmp/Lora_chatglm'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)