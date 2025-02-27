import time
import torch

from transformers import AutoTokenizer, AutoModel
# torch.set_default_tensor_type(torch.cuda.HalfTensor)


def inference(
        model,
        tokenizer,
        instuction: str,
        sentence: str
    ):
    """
    模型 inference 函数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        print(f'input_text1-->{input_text}')
        if sentence:
            input_text += f"Input: {sentence}\n"
        print(f'input_text2--》{input_text}')
        input_text += f"Answer: "
        print(f'input_text3--》{input_text}')
        batch = tokenizer(input_text, return_tensors="pt")
        print(f'batch--->{batch["input_ids"].shape}')
        out = model.generate(
            input_ids=batch["input_ids"].to(device),
            max_new_tokens=max_new_tokens,
            temperature=0
        )
        print(f'out-->{out}')
        out_text = tokenizer.decode(out[0])
        print(f'out_text-->{out_text}')
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    device = 'mps:0'
    max_new_tokens = 300
    model_path = "/Users/ligang/PycharmProjects/llm/ptune_chatglm/checkpoints/model_1800"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    ).half().to(device)

    samples = [
        {
            'instruction': "现在你是一个非常厉害的SPO抽取器。",
            "input": "下面这句中包含了哪些三元组，用json列表的形式回答，不要输出除json外的其他答案。\n\n73获奖记录人物评价：黄磊是一个特别幸运的演员，拍第一部戏就碰到了导演陈凯歌，而且在他的下一部电影《夜半歌声》中演对手戏的张国荣、吴倩莲、黎明等都是著名的港台演员。",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面子中的主语是什么类别，输出成列表形式。\n\n第N次入住了，就是方便去客户那里哈哈。还有啥说的"
        }
    ]

    start = time.time()
    for i, sample in enumerate(samples):
        print(f'sample-->{sample}')
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input']
        )
        print(f'res {i}: ')
        print(res)
    print(f'Used {round(time.time() - start, 2)}s.')