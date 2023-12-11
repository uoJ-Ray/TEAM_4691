import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default="./")
parser.add_argument('-output_file', type=str, default='tmp.tsv')

args = parser.parse_args()

def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()
    
def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案

    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, windows=0):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)

    bounary , item_idx , temp_seq , seq_pairs = 0 , 0 , "" , []
    new_line_idx = 0
    c = 0
    for w_idx, word in enumerate(article):
        if word == '\n':
            new_line_idx = w_idx + 1
            if article[bounary:new_line_idx] == '\n':
                c += 1
                continue
            if temp_seq == "":
                temp_seq = "PHI:NULL"
            sentence = article[bounary:new_line_idx].strip().replace('\t' , ' ')
            if len(sentence) == 0:
                c += 1
                temp_seq = ""
                continue
            if sentence[0] == '\"':
                sentence = sentence[1:]
            temp_seq = temp_seq.strip('\\n')
            seq_pair = f"{txt_name}\t{new_line_idx}\t{sentence}\t{temp_seq}\n"
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
            c += 1
            
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if "PHI:NULL" in temp_seq:
                print(1)
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\\n"
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    assert len(sents) == c
    return seq_pairs

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder, windows=0):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())
    print("processing each medical file")

    all_seq_pairs = []
    for txt_name in txt_names:
        all_seq_pairs.extend(process_medical_report(txt_name, medical_report_folder, annos_dict, windows))
    # all_labels = [i.split('\t')[-1].strip('\n') for i in all_seq_pairs]
    # fid = [i.split('\t')[0].strip('\n') for i in all_seq_pairs]
    # for i in tqdm(range(len(all_seq_pairs))):
    #     tmp = all_seq_pairs[i].split('\t')
    #     pre, aft = "", ""
    #     label = [all_labels[i]]
    #     if i > 0 and "PHI:NULL" != all_labels[i - 1] and fid[i - 1] == fid[i]:
    #         label = [all_labels[i - 1]] + label
    #     if i < len(all_seq_pairs) - 1 and "PHI:NULL" != all_labels[i + 1] and fid[i + 1] == fid[i]:
    #         label += [all_labels[i + 1]]
    #     label = "\\n".join([l for l in label if l != "PHI:NULL"])
    #     if label == "":
    #         label = "PHI:NULL"
    #     label += "\n"
    #     all_seq_pairs[i] = "\t".join(tmp[:-1] + [label])
    print("All medical file done")
    return all_seq_pairs

def main():
    anno_info_path = os.path.join(args.data_dir, "First_Phase_Release(Correction)/answer.txt")
    report_folder = os.path.join(args.data_dir, "First_Phase_Release(Correction)/First_Phase_Text_Dataset")
    first = generate_annotated_medical_report_parallel(anno_info_path, report_folder)
    anno_info_path = os.path.join(args.data_dir, "Second_Phase_Dataset/answer.txt")
    report_folder = os.path.join(args.data_dir, "Second_Phase_Dataset/Second_Phase_Text_Dataset")
    second = generate_annotated_medical_report_parallel(anno_info_path, report_folder)
    all_seq_pairs = first + second
    print(f"{len(all_seq_pairs)} training data")
    with open(args.output_file , 'w' , encoding = 'utf-8') as fw:
        for seq_pair in all_seq_pairs:
            fw.write(seq_pair)
    print("tsv format dataset done")
    
if __name__ == "__main__":
    main()