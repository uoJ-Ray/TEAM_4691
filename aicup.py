import random
import torch

class OpenDeidBatchSampler():    
    def __init__(self, data, batch_size, ids):
        self.pooled_indices = []
        self.data = data
        self.batch_size = batch_size
        self.ids = set(ids)
        self.len = len(list(data))  
    def __iter__(self):
        self.pooled_indices = []
        indices = [(index, len(data["content"])) for index, data in enumerate(self.data) if index in self.ids]
        random.shuffle(indices)
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), self.batch_size * 100):
            self.pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1], reverse=True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]

        # yield indices for current batch
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield self.pooled_indices[i:i + self.batch_size]
    def __len__(self):
        return (self.len + self.batch_size - 1) // self.batch_size
    
def collate_batch_with_prompt_template_t5(batch, tokenizer, template = "Extract the Protected Health Information (PHI) from below: __CONTENT__", IGNORED_PAD_IDX = -100):
    """ template: __CONTENT__ and __LABEL__ will be replaced with the content and the corresponding labels."""	
    # default template: {bos} {data['content']} {sep}
	
    batch = list(batch)
    for idx, data in enumerate(batch):
        label = [i for i in data['label'].split('\\n') if 'DATE' not in i and 'TIME' not in i and 'DURATION' not in i and 'SET:' not in i]
        label = "\\n".join(label)
        if label == "":
            label = "PHI:NULL"
        batch[idx]['label'] = label
        
    texts = [template.replace("__CONTENT__", data['content']) for data in list(batch)]
    encoded_seq = tokenizer(texts, padding="longest", max_length=512, truncation=True)
    indexed_tks = torch.tensor(encoded_seq['input_ids'])
    attention_mask = torch.tensor(encoded_seq['attention_mask'])
    labels = [data['label'] for data in list(batch)]
    # print('no date')
    # input('zzz')
    encoded_label = torch.tensor(tokenizer(labels, padding="longest", max_length=512, truncation=True)['input_ids'])
    encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX
    return indexed_tks, encoded_label, attention_mask

def collate_batch_with_prompt_template_t5_date(batch, tokenizer, template = "Extract the Protected Health Information (PHI) from below: __CONTENT__", IGNORED_PAD_IDX = -100):
    """ template: __CONTENT__ and __LABEL__ will be replaced with the content and the corresponding labels."""	
    # default template: {bos} {data['content']} {sep}
    
    batch = list(batch)
    texts = [template.replace("__CONTENT__", data['content']) for data in batch]
    encoded_seq = tokenizer(texts, padding="longest", max_length=512, truncation=True)
    indexed_tks = torch.tensor(encoded_seq['input_ids'])
    attention_mask = torch.tensor(encoded_seq['attention_mask'])
    labels = [data['label'] for data in list(batch)]
        
    encoded_label = torch.tensor(tokenizer(labels, padding="longest", max_length=512, truncation=True)['input_ids'])
    # print('date')
    # input('zzz')
    encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX
    
    return indexed_tks, encoded_label, attention_mask