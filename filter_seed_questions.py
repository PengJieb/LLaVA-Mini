import json

ori_ques = []
with open('playground/data/eval/seed_bench/llava-seed-bench.jsonl', 'r') as json_file:
    for line in json_file:
        if line.strip():  # skip empty lines
            ori_ques.append(json.loads(line))
print('ori len', len(ori_ques))

annos = json.load(open('playground/data/eval/seed_bench/SEED-Bench/SEED-Bench.json'))

anno_ques = annos['questions']
anno_ques_ids = [aq['question_id'] for aq in anno_ques]
print(len(anno_ques_ids))

new_ques = []
for oq in ori_ques:
    if str(oq['question_id']) in anno_ques_ids:
        new_ques.append(oq)
    elif 'v' not in oq['question_id'] and int(oq['question_id']) in anno_ques_ids:
        new_ques.append(oq)
    else:
        # print(oq['question_id'])
        pass
print(len(new_ques))

save_file = 'playground/data/eval/seed_bench/llava-seed-bench-filtered.jsonl'
with open(save_file, 'w', encoding='utf-8') as f:
    for item in new_ques:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')