import random

'''
对输入模型的指令进行增强：

0.15概率随机重复：
    每个字符0.1概率随机重复
    
0.15概率随机删除:
    每个字符0.1概率随机删除
'''

def random_repeat(s, max_repeats=2):
    if random.random() > 0.15:
        return s
    
    new_s = ""
    for char in s:
        if random.random() > 0.2:
            repeat = random.randint(1, max_repeats)
            new_s += char * repeat
        else:
            new_s += char
    # print(f'after aug:{new_s}')
    return new_s

def random_delete(s):
    if random.random() > 0.15:
        return s
    
    new_s = ""
    for char in s:
        if random.random() > 0.2:
            new_s += char
    # print(f'after aug:{new_s}')
    
    return new_s
