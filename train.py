import os
import time
import copy
import argparse
from functools import partial
import peft    #peft这个第三方库含有P-Tuning、LoRA等微调方法
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在CPU环境中不起任何作用
from torch.cuda.amp import autocast as autocast  #混合精度
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler, AutoModelForCausalLM, \
    BertForSequenceClassification
from utils.common_utils import *
from data_loader import *
from glm_config import *
#训练的损失如果出现nan，就换一个Adamw来源,不要使用pytorch的Adamw
# from transformers import AdamW


pc = ProjectConfig()

def evaluate_model(model, dev_dataloader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if pc.use_lora:
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


def model2train():
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)  #加载glm的配置文件
    # print(f"config---->{config}")
    #glm的配置文件是可以改的，下面是使用P-Tuning的走法
    if pc.use_ptuning:
        config.pre_seq_len = pc.pre_seq_len
        #指定是v1还是v2
        config.prefix_projection = pc.prefix_projection
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,
                                      trust_remote_code=True)

    #model.half()将模型数据类型从默认的float32精度转换为更低的float16精度，进一步减少内存
    model = model.float()   #更换精度
    print(model)

    #时间换空间，减少内存
    model.gradient_checkpointing_enable() # 梯度检查点是一种优化技术，用于在反向传播过程中降低内存使用
    model.enable_input_require_grads()  # 保存部分激活值，未保存的反向传播时重新计算

    model.config.use_cache = False    # 不进行缓存，减少内存。当需要某个“缓存值”的时候再重新计算，费时省内存
    #如果使用P-Tuning
    if pc.use_ptuning:
        model.transformer.prefix_encoder.float()   #输出限制为float  32位
    # print(f'model.lm_head-->{model.lm_head}')
    #如果使用LoRA
    if pc.use_lora:
        model.lm_head = CastOutputToFloat(model.lm_head) #确保模型输出头位32位，glm默认是32位，如果是其他模型就要有这一行   GPT的模型才需要这一行
 #------------------下面两行是使用bert模型
        # model = BertForSequenceClassification.from_pretrained(pc.pre_model, config=config)
        # #对于bert模型，不需要调整lm_heads
        # model.classifier = CastOutputToFloat(model.classifier)
  #----------------
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,  #使用常规的语言模型
            inference_mode=False, # 推理时为True，如果训练的时候则一般为False，决定是否使用dropout
            r=pc.lora_rank, # 低秩矩阵维度  一般为8
            lora_alpha=32, # 缩放系数
            lora_dropout=0.1,
        )
        model = peft.get_peft_model(model, peft_config)  #前面lora配置好后，直接这一样就能将lora配置加载到模型中，这样就相当于给模型加了一个旁路

    print(f'model2-->{model}')
    model = model.to(pc.device)
    print('模型训练参数', model.print_trainable_parameters()) #添加lora后的训练参数大概是3万多，glm-6b原始模型是60亿参数，相当于新增5%的参数进行微调

    no_decay = ["bias", "LayerNorm.weight"]  #一般大的模型都需要正则化，使用权重衰减，防止过拟合，但是偏置和层归一化除外
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    #如果训练损失为nan，就用transformers的Adamw
    # optimizer = AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)

    # model.to(pc.device)
    #
    train_dataloader, dev_dataloader = get_data()
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    #指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps) # 预热阶段的训练步数
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    #定义训练的一些参数变量
    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float('inf')
    for epoch in range(1, pc.epochs + 1):
        print("开始训练")

        for batch in train_dataloader:
            # print(batch['input_ids'].shape)
            # print(batch['labels'].shape)
            # labels = batch['labels'].view(-1)  # 将标签展平为 [batch_size]
            # if pc.use_lora:
            #     with autocast():
            #         loss = model(
            #             input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
            #             labels=batch['labels'].view(-1).to(dtype=torch.long, device=pc.device)  # 确保标签是 [batch_size]
            #         ).loss
            # else:
            #     loss = model(
            #         input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
            #         labels=batch['labels'].view(-1).to(dtype=torch.long, device=pc.device)  # 确保标签是 [batch_size]
            #     ).loss
            # print(batch['input_ids'].shape)  #torch.size([1,300])
            # print(batch['labels'].shape)   #torch.size([1,300])
            if pc.use_lora:
                # torch.cuda.amp.autocast是PyTorch中一种混合精度的技术（仅在GPU上训练时可使用）
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(float(loss.cpu().detach()))  #同一个批次的损失要存起来，后面计算平均损失

            global_step += 1
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print("global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s"
                      % (
                          global_step,
                          global_step / max_train_steps * 100,
                          epoch,
                          loss_avg,
                          pc.logging_steps / time_diff,
                          second2time(int(max_train_steps - global_step) / (pc.logging_steps / time_diff))
                      ))
                tic_train = time.time()
            #下面不再频繁保存模型，内存不够
            # if global_step % pc.save_freq == 0:
            #     cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
            #     save_model(model, cur_save_dir)
            #     tokenizer.save_pretrained(cur_save_dir)
            #     print(f'Model has saved at {cur_save_dir}.')
            #
            #     eval_loss = evaluate_model(model, dev_dataloader)
            #
            #     print("Evaluation Loss: %.5f" % (eval_loss))
            #
            #     if eval_loss < best_eval_loss:
            #         print(
            #             f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}"
            #         )
            #         best_eval_loss = eval_loss
            #         cur_save_dir = os.path.join(pc.save_dir, "model_best")
            #         save_model(model, cur_save_dir)
            #         tokenizer.save_pretrained(cur_save_dir)
            #         print(f'Best model has saved at {cur_save_dir}.')
        #一个epoch结束再保存模型
        cur_save_dir = os.path.join(pc.save_dir, str(epoch))
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        save_model(model, cur_save_dir)
        tokenizer.save_pretrained(cur_save_dir)
        print('model has save at {cur_save_dir}')

                # tic_train = time.time()


if __name__ == '__main__':
    model2train()