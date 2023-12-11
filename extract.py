import os
import re
import pickle as pkl
from argparse import ArgumentParser
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument('-dataset', type=str, default='./opendid_test')
parser.add_argument('-infile', type=str, default='./test_t5_only.txt')
parser.add_argument('-infile_date', type=str, default='./test_outputs_date.txt')
parser.add_argument('-output_file', type=str, default='./final2.txt')
args = parser.parse_args()

def get_content():
    filename = os.listdir(args.dataset)
    testcontents = {}
    print(len(filename))
    for file in filename:
        idx = file.split('.')[0]
        with open(os.path.join(args.dataset, file), "r") as f:
            testcontents[idx] = f.read()
    return testcontents

contents = get_content()

def extract_patient(outputs):
    ret = []
    for k, v in contents.items():
        names = re.findall("(?:LastName|MiddleName|FirstName)\n\w+\n", v)
        for name in names:
            name = name.split('\n')[1].strip()
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tPATIENT\t{st}\t{en}\t{name}"
            ret.append(s)
    for i in outputs:
        fid, cat, st, en, name = i.split('\t')
        i = "\t".join(i)
        if cat == "MEDICALRECORD" and fid.startswith("file") == False:
            name = contents[fid][int(en):].strip().split('\n')[0].strip()
            st = contents[fid].index(name)
            en = st + len(name)
            s = f"{fid}\tPATIENT\t{st}\t{en}\t{name}"
            ret.append(s)
    print(f"Extract {len(ret)} PATIENT")
    return ret

def extract_street():
    ret = []
    for k, v in contents.items():
        names = re.findall("Lab\s*No:.*\n.+\n", v)
        for name in names:
            name = name.split('\n')[1].strip()
            if 'BOX' in name:
                continue
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tSTREET\t{st}\t{en}\t{name}"
            ret.append(s)
    print(f"Extract {len(ret)} STREET")
    return ret

def extract_org():
    ret = []
    candidate = {'BlueLinx', 'The Pepsi', 'Boeing', 'Hormel Foods', 'H. J. Heinz', 'IKON Office Solutions', 'UnumProvident', 'Jones Apparel Group', 'The ServiceMaster Company', 'YRC Worldwide', 'LandAmerica Financial', 'Goldman Sachs', 'Yahoo', 'Valero Energy', 'Phelps Dodge', 'eBay', 'Johnson & Johnson', 'Leggett & Platt', 'Lear Corporation', 'Washington Mutual', 'Fifth Third Bancorp', 'Health Net', 'CHS Inc', 'Ashland', 'Avaya', 'Boston Scientific', 'Smithfield Foods', 'PepsiCo', 'Computer Sciences Corporatio', 'Entergy Corporation', 'The Northwestern Mutual', 'Schering-Plough', 'Henry Schein', "Albertson's", 'Micron Technology', 'Terex Corporation', 'Rohm and Haas', "Pilgrim's Pride", 'BorgWarner', 'Baker Hughes Incorporated', 'Express Scripts', 'State Street Corporation', 'Owens & Minor', 'Group 1 Automotive', 'ToysRus', 'Alcoa Inc', 'Hershey Company', 'BB&T Corporation', 'KB Home Los Angeles', 'W.W. Grainger Inc', 'YUM! Brands', 'Pacific Mutual', 'Federal-Mogul', 'McGraw-Hill', 'Textron Inc', 'Saks Incorporated', 'AES Corporation', 'The Timken Company', 'Peter Kiewit Sons', 'Caremark Rx', 'HCA Inc', 'Brunswick Corporation', 'CSX Corporation', 'CIT Group', 'Lyondell Chemical', 'Wachovia Corporatio', 'Coventry Health Care', 'Thrivent Financial', 'Lubrizol Corporation', 'Sonic Automotive', 'M.D.C. Holdings', 'AutoZone', 'CMS Energy Corporatio', 'Computer Sciences Corporation', 'Dollar General', 'United Auto', 'TransMontaigne', 'Delphi', 'Atmos Energy', 'Visteon Corporation', 'Molson Coors Brewing', 'Lucent Technologies', 'Ameren Corporatio', 'Weyerhaeuser Company', 'WPS Resources', 'Kimberly-Clark', 'Coca-Cola', 'Newell Rubbermaid', 'Hexion Specialty Chemicals', 'Wachovia Corporation', 'Coca Cola', 'Home Depot', 'R.R. Donnelley', 'Sealed Air Corporation', 'Burlington Northern Santa Fe', 'Devon Energy', 'NIKE', 'Estee Lauder', 'FPL Group', 'Avery Dennison', 'AmerisourceBergen Corporation', 'Sun Microsystems', 'ALLTEL Corporation', 'Liz Claiborne', 'Liberty Mutual', 'Fortune Brands', 'Wells Fargo', 'The First American', 'KeyCorp', 'Sanmina-SCI', 'Agilent Technologies', 'OGE Energy', 'Delta Air Lines', 'Walt Disney', "Dillard's"}
            
    for k, v in contents.items():
        for cand in candidate:
            if cand in v:
                name = cand
                st = v.index(name)
                en = st + len(name)
                s = f"{k}\tORGANIZATION\t{st}\t{en}\t{name}"
                ret.append(s)  
    print(f"Extract {len(ret)} ORGANIZATION")
    return ret

def extract_doctor():
    with open(args.infile_date, "r", encoding="utf8") as f:
        ret = f.readlines()
    ret = [i.strip('\n') for i in ret if i.split('\t')[1] == 'DOCTOR']
    print(f"Extract {len(ret)} DOCTOR")
    return ret

def extract_loc():
    ret = []
    for k, v in contents.items():
        names = re.findall("P\.?\s*O\.?\s*BOX\s*\d+", v)
        for name in names:
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tLOCATION-OTHER\t{st}\t{en}\t{name}"
            ret.append(s) 
    print(f"Extract {len(ret)} LOCATION")
    return ret

def extract_med():
    ret = []
    for k, v in contents.items():
        names = re.findall("\d+\.[A-Z][A-Z][A-Z]", v)
        for name in names:
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tMEDICALRECORD\t{st}\t{en}\t{name}"
            ret.append(s)  
        names = re.findall("MRN no:\s*\d+\n", v)
        for name in names:
            name = name.split(':')[1].strip()
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tMEDICALRECORD\t{st}\t{en}\t{name}"
            ret.append(s) 
        names = re.findall("MRN\n\d+\n", v)
        for name in names:
            name = name.split('\n')[1]
            st = v.index(name)
            en = st + len(name)
            s = f"{k}\tMEDICALRECORD\t{st}\t{en}\t{name}"
            # print(s)
            ret.append(s)
    print(f"Extract {len(ret)} MEDICALRECORD")
    return ret

def extract_age():
    ret = []
    for k, v in contents.items():
        names = re.findall("\d+ year old", v)
        for name in names:
            st = v.index(name)
            name = name.split()[0]
            en = st + len(name)
            s = f"{k}\tAGE\t{st}\t{en}\t{name}"
            ret.append(s)  
        names = re.findall(" age \d+", v)
        for name in names:
            st = v.index(name) + 5
            name = name.split('age ')[1]
            en = st + len(name)
            s = f"{k}\tAGE\t{st}\t{en}\t{name}"
            ret.append(s)
        names = re.findall("CLINICAL:\n\d+", v)
        for name in names:
            st = v.index(name) + len("CLINICAL:\n")
            name = name.split('\n')[1]
            if len(name) == 1:
                continue
            en = st + len(name)
            s = f"{k}\tAGE\t{st}\t{en}\t{name}"
            ret.append(s)
    print(f"Extract {len(ret)} AGE")
    return ret

def extract(outputs):
    ret = extract_patient(outputs)
    ret += extract_street()
    ret += extract_org()
    ret += extract_doctor()
    ret += extract_age()
    ret += extract_loc()
    ret += extract_med()
    
    tmp = set()
    outputs = [i for i in outputs if i.split('\t')[1] != 'STREET']
    tmp = list(set(outputs + ret))
    print(f"Extract {len(tmp)} data")
    return list(tmp)

def main():
    with open(args.infile, "r") as f:
        outputs = f.readlines()
    outputs = [i.strip() for i in outputs]
    total_outputs = extract(outputs)
    outputs = [i for i in outputs if i.split('\t')[1] not in ['PATIENT', 'STREET', 'ORGANIZATION', 'AGE', 'LOCATION-OTHER', 'MEDICALRECORD']]
    outputs += total_outputs
    print(f"Extract {len(total_outputs)} entities for task 1")
    print(f"Write the outputs to the {args.output_file}")
    with open(args.output_file, "w") as f:
        for i in total_outputs:
            f.write(i + '\n')
    
if __name__ == "__main__":
    main()