import os
import re
from argparse import ArgumentParser
from datasets import load_dataset, Features, Value

parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='./opendid_test')
parser.add_argument('-dataset', type=str, default='./opendid_test.tsv')
parser.add_argument('-output_file', type=str, default='./rule.txt')
args = parser.parse_args()

month = {"January":"01", "February":"02", "March":"03", "April":"04", "May":"05", "June":"06", "July":"07", "August":"08",
        "September":"09", "October":"10", "November":"11", "December":"12"}
keys = list(month.keys())
for k in keys:
    month[k.upper()] = month[k]
    month[k.lower()] = month[k]   
keys = set(['DATE' , "TIME" , "DURATION" , "SET"])

def get_content():
    filename = os.listdir(args.data_dir)
    testcontents = {}
    print(len(filename))
    for file in filename:
        idx = file.split('.')[0]
        with open(os.path.join(args.data_dir, file), "r") as f:
            testcontents[idx] = f.read()
    return testcontents

contents = get_content()

def m2n(s):
    for k in month.keys():
        if k in s:
            return k, month[k]
    return None

def check(s):
    k = ["on", "at", "am", "a.m.", "a.m", "p.m.", "pm", "p.m"]
    for i in k:
        if i in s:
            return False
    return True

def find_datetime(test_list):
    seqs = []
    for idx, ex in enumerate(test_list):
        st = 0
        ans = ""
        while st < len(ex['content']):
            content = ex['content'][st:]
            fid = ex['fid']
            pattern = ["\d+[.\-/]\d+[.\-/]*\d*\w*\s(?:on|at|on the|@)?\s*\d+[:.]*\d+[:.]*\d*\s?[hrsapm.]*", "\d+[:.]*\d*[:.]*\d*\s?[hrsapm.]*\s*(?:and|on|at|on the)?\s*\d+[.\-/]\d+[.\-/]*\d*"]
            pattern += ["\d+[.\-/]\d+[.\-/]*\d*", "(?:on|at|on the)+\s*\d+[:.]\d+[:.]*\d*\s?[hrsapm.]*"]
            pattern += [f"\d+[:.]\d+[:.]*\d*\s?[hrsapm.]* on the\s?\d+(?:st|nd|rd|th)?\s?(?:of)?\s?(?:{'|'.join(list(month.keys()))})? \d+", f"\d+(?:st|nd|rd|th)? (?:{'|'.join(list(month.keys()))})? \d+ at \d+[:.]\d+[:.\d]*\s?[hrsapm.]*\s*"]
            pattern += [f"\d+Hrs (?:on|on the)? \d+[.\-/]\d+[.\-/]*\d*", f"\d+[.\-/]\d+[.\-/]*\d* (?:at)? \d+\s*Hrs"]
            found = []
            for n, p in enumerate(pattern):
                ret = re.findall(p, content)
                ret = [(n, i.strip()) for i in ret]
                found.extend(ret)
            if len(found) != 0:
                num, s = max(found, key=lambda x:(len(x[1]), -x[0]))
                st += content.index(s) + len(s)
                s = s.replace("m.", "m")
                if num == 0 and s[-1] == '.' and 'p.m.' not in s and 'a.m.' not in s and 'pm.' not in s and 'am.' not in s:
                    s = s[:-1]
                if s[0] == '-' or s[-1] == '-' or s[-1] == '.' or 'mm' in s or bool(re.search("\d\.\d\.\d\.\d", s)):
                    # input('b')
                    continue
                if check(s) and (len(s.split('.')) == 2 or len(s.split('/')) == 2 or len(s.split('-')) == 2):
                    continue
                s = s.strip(":")
                start = ex['idx'] + ex['content'].index(s)
                end = start + len(s)
                extract_s = s
                
                if " pm" in s:
                    s = s.replace(" pm", "pm")
                elif " p.m." in s:
                    s = s.replace(" p.m", "pm")
                elif " H" in s:
                    s = s.replace(" H", "H")
                if num == 4 or num == 5:
                    if num == 4:
                        tmp_s = s.replace("on the", " ")
                        tmp_s = " ".join(tmp_s.split())
                    time = re.findall("\d+[:.]\d+[:.\d]*\s?[hrsapm.]*", tmp_s)[0]
                    day = re.findall("\d+(?:st|nd|rd|th)", tmp_s)[0]
                    day = re.sub('[a-zA-Z]+', '', day).zfill(2)
                    k, m = m2n(tmp_s)
                    if num == 4:
                        y = tmp_s.strip().split()[-1]
                    elif num == 5 or num == 6:
                        y = tmp_s[tmp_s.strip().index(k) + len(k) + 1:].split()[0]                        
                    date = f"{y}-{m}-{day}"
                else:
                    if s.split()[0].isalpha():
                        s = " ".join(s.split()[1:])
                    split_s = s.split()
                    # print(split_s)
                    date, time = split_s[0], split_s[-1]

                    if 'on' in s or ':' in date or "pm" in date or "am" in date:
                        date, time = time, date
                if date[0] == '/':
                    date = date[1:]
                if date.isalpha() or (len(split_s) == 1 and (':' in date or "pm" in date or "am" in date)):
                    date = ""
                time = time.replace('p.m.', "pm").replace("p.m", "pm").replace('a.m.', "am").replace("a.m", "am")
                if '/' in time or '-' in time or ('.' in time and time.split('.')[-1].isnumeric() and len(time.split('.')[-1]) == 4) or ('.' in time and len(time.split('.')) == 3):
                    time = ""
                if '.' in date and len(date.split('.')[0]) == 4:
                    continue
                if '-' in date and len(date.split('-')[0]) != 4:
                    continue
                if '.' in date:
                    datelist = date.split('.')[::-1]
                elif '/' in date:
                    datelist = date.split('/')[::-1]
                if '.' in date or '/' in date:   
                    datelist = [re.sub('[a-zA-Z]+', '', i) for i in datelist]
                    if len(datelist[0]) > 4:
                        datelist[0] = datelist[0][:4]
                    for i in range(len(datelist)):
                        if len(datelist[i]) < 2:
                            datelist[i] = '0' + datelist[i]

                    date = "-".join(datelist)
                    if date[2] == '-':
                        date = '20' + date
                # print(date)
                # print(time)
                if time.isalpha():
                    time = ""
                if time[-2:] == "m.":
                    time = time[:-1]
                # print(time)
                time = time.replace('.', ':')
                if 'pm' in time:
                    if ':' in time:
                        t = time.split(':')
                    else:
                        t = [re.sub('[^0-9]', '', time), '00']
                    if int(t[0]) < 12:
                        time = ":".join([str(int(t[0]) + 12)] + t[1:])
                    else:
                        time = ":".join(t)
                time = re.sub('[^0-9:.]', '', time)
                if ':' not in time:
                    time = ":".join(time[i:i+2] for i in range(0, len(time), 2))
                if time != "":
                    time = ":".join([i.zfill(2) for i in time.split(':')])
                if time != "" and ':' not in time:
                    time = time + ':00'
                # print(time)
                if date == "":
                    ans = f"TIME:{s}=>{time}"
                    sec = "TIME"
                elif time == "":
                    if s[-2:] == " a":
                        s = s[:-2]
                    ans = f"DATE:{s}=>{date}"
                    sec = "DATE"
                else:
                    ans = f"TIME:{s}=>{date}T{time}"
                    sec = "TIME"
                seqs.append(f"{fid}\t{sec}\t{start}\t{end}\t{extract_s}\t{ans.split('=>')[1]}")
            else:
                break
    for k, v in contents.items():
        birth = re.findall("DateOfBirth\n\d+", v)
        for b in birth:
            # print(b)
            date = b.split('\n')[1][:8]
            st = v.index(date)
            en = st + len(date)
            norm_date = date[:4] + '-' + date[4:6] + '-' + date[-2:]
            seqs.append(f"{k}\tDATE\t{st}\t{en}\t{date}\t{norm_date}")
            # print(seqs[-1])
    return seqs

def find_duration(test_list):
    seqs = []
    for idx, i in enumerate(test_list):
        content = i['content']
        fid = i['fid']
        boundary = i['idx']
        pattern = ["\d*-*\d*/*\s*/*(?:weeks|wks|week|months|month|yrs|years|year|yr)\s*"]
        ret = re.findall(pattern[0], content)
        ans = ""
        if len(ret) != 0:
            for s in ret:
                s = s.strip()
                start = boundary + content.index(s)
                end = start + len(s)
                if s.isalpha() or s + ' old' in content or s + '-old' in content:
                    continue
                if 'weeks' in s or 'wks' in s or 'week' in s:
                    num = re.sub('[a-zA-Z]+', '', s).strip()
                    if '-' in num:
                        num = [int(i) for i in num.split('-')]
                        num = round(sum(num) / len(num), 2)
                    ans += f"\nDURATION:{s}=>P{num}W"
                    l = f"P{num}W"
                elif 'month' in s or 'months' in s:
                    num = re.sub('[a-zA-Z]+', '', s).strip()
                    if '-' in num:
                        num = [int(i) for i in num.split('-')]
                        num = round(sum(num) / len(num), 2)
                    ans += f"\nDURATION:{s}=>P{num}M"
                    l = f"P{num}M"
                elif 'yrs' in s or 'years' in s or 'year' in s or 'yr' in s:
                    if 'CLINICAL:\n' in test_list[idx-1]['content']:
                        continue
                    num = re.sub('[a-zA-Z]+', '', s).strip()
                    if '-' in num:
                        num = [int(i) for i in num.split('-')]
                        num = round(sum(num) / len(num), 2)
                    ans += f"\nDURATION:{s}=>P{num}Y"
                    l = f"P{num}Y"
                if l != "PY":
                    seqs.append(f"{fid}\tDURATION\t{start}\t{end}\t{s}\t{l}")
        if 'twice' in content:
            ans += "\nSET:twice=>R2"
            start = boundary + content.index('twice')
            end = start + 5
            seqs.append(f"{fid}\tSET\t{start}\t{end}\ttwice\tR2")
    return seqs

def main():
    test_data = load_dataset("csv", data_files=args.dataset, delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
    test_list= list(test_data['train'])
    test_list = [i for i in test_list if i['content'] != None]
    datetime_seq = find_datetime(test_list)
    print(f"Found {len(datetime_seq)} dates & times in data")
    duration_seq = find_duration(test_list)
    print(f"Found {len(duration_seq)} durations in data")
    total = datetime_seq + duration_seq
    print(f"Write the outputs to the {args.output_file}")
    with open(args.output_file, "w", encoding="utf8") as f:
        for i in total:
            f.write(i + '\n')
            
if __name__ == "__main__":
    main()