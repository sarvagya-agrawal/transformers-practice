from pathlib import Path

train_lines = [line.strip() for line in Path(
    'resources/gpt2/train.history_belief').open().readlines()]
valid_lines = [line.strip() for line in Path(
    'resources/gpt2/val.history_belief').open().readlines()]
test_lines = [line.strip() for line in Path(
    'resources/gpt2/test.history_belief').open().readlines()]

train_src = list()
train_tgt = list()
valid_src = list()
valid_tgt = list()
test_src = list()
test_tgt = list()

context = 2
out_dir = 'mw-data'
Path(out_dir).mkdir(exist_ok=True)
for line in train_lines:
    line = line.replace('<|endoftext|>', '').replace('<|endofcontext|>', '').split(
        '<|context|>')[1].replace('<|endofbelief|>', '').split('<|belief|>')
    src = line[0].replace('<|user|>', '__User').replace(
        '<|system|>', '__Agent').strip() + ' __Belief\n'
    if src.count('__User') > context:
        src = '__User ' + '__User '.join(src.split('__User')[-context:])
    train_src.append(src.replace('  ', ' '))
    slots = line[1].strip().split(',')
    slots = [
        slot.strip() for slot in slots if 'not mentioned' not in slot or 'none' not in slot]
    # slots = [slot.strip() for slot in slots]
    # skip = False
    # for slot in slots:
    #     if 'taxi' == slot.strip().split(' ')[0]:
    #         del train_src[-1]
    #         skip = True
    #         break
    # if skip:
    #     continue
    s = '< ' + (' , ').join(slots) + ' >\n'
    train_tgt.append(s.replace('  ', ' '))

for line in valid_lines:
    line = line.replace('<|endoftext|>', '').replace('<|endofcontext|>', '').split(
        '<|context|>')[1].replace('<|endofbelief|>', '').split('<|belief|>')
    src = line[0].replace('<|user|>', '__User').replace(
        '<|system|>', '__Agent').strip() + ' __Belief\n'
    if src.count('__User') > context:
        src = '__User ' + '__User '.join(src.split('__User')[-context:])
    valid_src.append(src.replace('  ', ' '))
    slots = line[1].strip().split(',')
    slots = [
        slot.strip() for slot in slots if 'not mentioned' not in slot or 'none' not in slot]
    # slots = [slot.strip() for slot in slots]
    # skip = list()
    # for slot in slots:
    #     if 'taxi' != slot.strip().split(' ')[0]:
    #         skip.append(True)
    #         continue
    #     skip.append(False)
    # if all(skip):
    #     del valid_src[-1]
    #     continue
    s = '< ' + (' , ').join(slots) + ' >\n'
    valid_tgt.append(s.replace('  ', ' '))

for line in test_lines:
    line = line.replace('<|endoftext|>', '').replace('<|endofcontext|>', '').split(
        '<|context|>')[1].replace('<|endofbelief|>', '').split('<|belief|>')
    src = line[0].replace('<|user|>', '__User').replace(
        '<|system|>', '__Agent').strip() + ' __Belief\n'
    if src.count('__User') > context:
        src = '__User ' + '__User '.join(src.split('__User')[-context:])
    test_src.append(src.replace('  ', ' '))
    slots = line[1].strip().split(',')
    slots = [
        slot.strip() for slot in slots if 'not mentioned' not in slot or 'none' not in slot]
    # slots = [slot.strip() for slot in slots]
    # skip = list()
    # for slot in slots:
    #     if 'taxi' != slot.strip().split(' ')[0]:
    #         skip.append(True)
    #         continue
    #     skip.append(False)
    # if all(skip):
    #     del test_src[-1]
    #     continue
    s = '< ' + (' , ').join(slots) + ' >\n'
    test_tgt.append(s.replace('  ', ' '))

with open(f'{out_dir}/train.src', 'w') as f:
    for line in train_src:
        f.write(line)
with open(f'{out_dir}/train.tgt', 'w') as f:
    for line in train_tgt:
        f.write(line)
with open(f'{out_dir}/valid.src', 'w') as f:
    for line in valid_src:
        f.write(line)
with open(f'{out_dir}/valid.tgt', 'w') as f:
    for line in valid_tgt:
        f.write(line)
with open(f'{out_dir}/test.src', 'w') as f:
    for line in test_src:
        f.write(line)
with open(f'{out_dir}/test.tgt', 'w') as f:
    for line in test_tgt:
        f.write(line)
