import os
import pickle as pkl
from argparse import ArgumentParser
from openai import OpenAI
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument('-api_key', type=str, default='')
parser.add_argument('-test_dataset', type=str, default='./opendid_test')
args = parser.parse_args()

## Few-Shots prompt for gpt
p = ["Episode No:  88Y206206L\n8892062.BPL\n\nVatterott, Jerrie CLARENCE \nLab No:  88Y20620,88Y20620\nExeter\nD", "Vatterott, Jerrie CLARENCE"]
p += ['TRICT HOSPITAL Number c33J47905-G0CRene. \nClinical notes: pT2 N3a invasive duct Ca.\nHER2 IHC Result: 1+ at WANTIRNA HEALTH.\nSignal Detection has been performed for the HER2 gene (17q21) andchromosome 17 centromeric enumeration probe (CEP1 7) control using the Ventana INFORM HER2 Dual ISH DNA Probe ', 'CRene']
p += ['2\n\n\nMRN\n783488\n\n\nSpecimenType\nFresh Tissue\n\n\nGender\nF\n\n\nMiddleName\nReazer\n\n\nFacilityID\nMR\n', 'Reazer']
p += [' metastasis\n- Sessile serrated adenoma in appendix.\n- pT3     pN1a     pMx     R0\t (TNM 8th)\n\n\n\n\n\nSpecimenReceivedDate\n2761-04-09 00:00:00\n\n\nLastName\nADRIAN-Dattilo\n', 'ADRIAN-Dattilo']

def get_content():
    filename = os.listdir(args.test_dataset)
    testcontents = {}
    print(len(filename))
    for file in filename:
        idx = file.split('.')[0]
        with open(os.path.join(args.test_dataset, file), "r") as f:
            testcontents[idx] = f.read()
    return testcontents

def main():
    contents = get_content()
    print("Connecting to OpenAI")
    client = OpenAI(api_key=args.api_key)
    system_prompt = "You are a helpful assistant. Please extract the patient's name from the passage and output the answer line by line.\n"
    system_prompt += "Each line should contain the patient's name."
    system_prompt += "If there is no any patient's name in the passage, please output 'NULL' for your answer\n"
    system_prompt += "Don\'t output anything else but only the value. Below shows some examples.\n"
    prompt_q = f"Passage: {p[0]}\n"
    prompt_a = "Answer: {p[1]}\n"
    prompt_q2 = f"Passage: {p[2]}\n"
    prompt_a2 = f"Answer: {p[3]}\n"
    prompt_q3 = f"Passage: {p[4][:90]}\n"
    prompt_a3 = f"Answer: {p[5][:6]}\n"
    prompt_q4 = f"Passage: {p[6]}\n"
    prompt_a4 = f"Answer: {p[7]}\n"
    chatout = []
    c = 0
    for k, v in tqdm(contents.items(), total=len(contents)):
        c += 1
        for l in range(0, len(v), 4096):
            prompt_q5 = f"Passage: {contents[k][l:l+4096]}\nAnswer:"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{prompt_q}"},
                    {"role": "assistant", "content": f"{prompt_a}"},
                    {"role": "user", "content": f"{prompt_q2}"},
                    {"role": "assistant", "content": f"{prompt_a2}"},
                    {"role": "user", "content": f"{prompt_q3}"},
                    {"role": "assistant", "content": f"{prompt_a3}"},
                    {"role": "user", "content": f"{prompt_q4}"},
                    {"role": "assistant", "content": f"{prompt_a4}"},
                    {"role": "user", "content": f"{prompt_q5}"}
                ]
            )
            out = response.choices[0].message.content
            chatout.append([k, out])
    tmp = []
    for file, ans in chatout:
        sections = ans.replace('Answer: ', '').split('\n')
        sections = [i for i in sections if i != ""]
        for ans in sections:
            if 'NULL' not in ans:
                name = ans.strip()
                if name in contents[file]:
                    st = contents[file].index(name)
                    en = st + len(name)
                    s = f"{file}\tPATIENT\t{st}\t{en}\t{name}"
                    tmp.append(s)
    print(f"Extract {len(tmp)} patient's name from GPT-3.5")
    print("Dumping these information into patient.pkl")
    with open("patient.pkl", "rb") as f:
        pkl.dump(tmp, f)
        
if __name__ == "__main__":
    main()
