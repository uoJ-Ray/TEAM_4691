import os
import re
import pickle as pkl
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from aicup import collate_batch_with_prompt_template_t5, collate_batch_with_prompt_template_t5_date, OpenDeidBatchSampler
from transformers import T5ForConditionalGeneration

train_phi_category = ['PATIENT', 'DOCTOR', 'USERNAME',
                    'PROFESSION',
                    'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                    'AGE',
                    'DATE', 'TIME', 'DURATION', 'SET',
                    'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
                    'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']
 
class Trainer(object):
    def __init__(self, device, config, tokenizer):
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(config.plm).to(device)
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")
        
    def train(self, dataset):
        dataset = list(dataset['train'])
        print(f"Total {len(dataset)} in  training datset!")
        train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, random_state=1110)
        kfold = KFold(n_splits=5, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=self.config.early_stop, verbose=True)
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        self.model, optimizer, scheduler = accelerator.prepare(self.model, optimizer, scheduler)
        all_train_loss = []
        all_val_loss = []
        print("Start Training ...")
        print(f"lr = {optimizer.param_groups[0]['lr']}")
        epoch = 0
        best_loss = float('inf')
        while True:
            epoch += 1
            step = 0
            total_loss = 0
            if self.config.date:
                train_dataloader = DataLoader(train_dataset, collate_fn=lambda batch: collate_batch_with_prompt_template_t5_date(batch, self.tokenizer),
                                            batch_sampler=OpenDeidBatchSampler(train_dataset, self.config.batch_size, [i for i in range(len(train_dataset))]), pin_memory=True)
                valid_dataloader = DataLoader(valid_dataset, collate_fn=lambda batch: collate_batch_with_prompt_template_t5_date(batch, self.tokenizer),
                                            batch_sampler=OpenDeidBatchSampler(valid_dataset, self.config.batch_size, [i for i in range(len(valid_dataset))]), pin_memory=True)
            else:
                train_dataloader = DataLoader(train_dataset, collate_fn=lambda batch: collate_batch_with_prompt_template_t5(batch, self.tokenizer),
                                            batch_sampler=OpenDeidBatchSampler(train_dataset, self.config.batch_size, [i for i in range(len(train_dataset))]), pin_memory=True)
                valid_dataloader = DataLoader(valid_dataset, collate_fn=lambda batch: collate_batch_with_prompt_template_t5(batch, self.tokenizer),
                                            batch_sampler=OpenDeidBatchSampler(valid_dataset, self.config.batch_size, [i for i in range(len(valid_dataset))]), pin_memory=True)
            total_step, total_valid_step = len(train_dataloader), len(valid_dataloader)

            self.model.train()
            train_dataloader, valid_dataloader = accelerator.prepare(train_dataloader, valid_dataloader)
            train_loss = 0
            for idx, (seqs, labels, masks) in tqdm(enumerate(train_dataloader), total=total_step):
                with accelerator.accumulate(self.model):
                    seqs = seqs.to(self.device)
                    labels = labels.to(self.device)
                    masks = masks.to(self.device)
                    self.model.zero_grad()
                    outputs = self.model(seqs, labels=labels, attention_mask=masks)
                    loss = outputs.loss.mean()
                    train_loss += loss.item()
                    if idx % 100 == 0:
                        all_train_loss.append(train_loss)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    del seqs, labels, masks, outputs
                    torch.cuda.empty_cache()
                            
            avg_train_loss = train_loss / total_step
            print(f"Epoch {epoch} | loss = {avg_train_loss:.3f} | lr = {optimizer.param_groups[0]['lr']}")
            # ---------- Validation ----------
            print("Evaluating Dev Set ...")
            dev_loss = 0
            self.model.eval()
            with torch.no_grad():
                for idx, (seqs, labels, masks) in tqdm(enumerate(valid_dataloader), total=total_valid_step):
                    seqs = seqs.to(self.device)
                    labels = labels.to(self.device)
                    masks = masks.to(self.device)
                    outputs = self.model(seqs, labels=labels, attention_mask=masks)
                    dev_loss += outputs.loss.mean().item()
                    del seqs, labels, masks, outputs
                    torch.cuda.empty_cache()
            dev_loss = dev_loss / total_valid_step
            all_val_loss.append(dev_loss)
            print(f"Validation | Epoch {epoch} | loss = {dev_loss:.3f}")
            total_loss += dev_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = epoch
                early_stop_count = 0
                self.model.save_pretrained(self.config.model_path)
                print(f'saving model {self.config.model_path} with loss {best_loss}')
            else:
                early_stop_count += 1
            scheduler.step(dev_loss)

            if early_stop_count > self.config.early_stop:
                early_stop_count = 0
                print(f"Reload Model {self.config.model_path}")
                self.model = self.model.from_pretrained(self.config.model_path).to(self.device)
                
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print('\nModel is not improving, so we halt the training session.')
                print(f'Best epoch in {best_epoch}, Best valid loss: {best_loss}')
                with open("loss_date.pkl", "wb") as f:
                    pkl.dump([all_train_loss, all_val_loss], f)
                return
               
    def aicup_predict(self, inputs, template = "Extract the Protected Health Information (PHI) from below: __CONTENT__"):
        def get_anno_format(sentence , infos , boundary):
            anno_list = []
            lines = infos.split("\n")
            normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
            phi_dict = {}
            tmp = set()
            for line in lines:
                sections = line.split("\\n")
                for sec in sections:
                    phi_dict = {}
                    sec = sec.strip()
                    sec = sec.replace("PHI:NULL", "")
                    norm_ext = 0
                    for k in normalize_keys:
                        if f'{k}:' in sec:
                            norm_ext = 1
                            parts = [f'{k}', sec[sec.index(f'{k}:') + len(k) + 1:]]
                            break
                    if norm_ext == 0:
                        parts = sec.split(":")
                        if len(parts) == 1 or parts[0] not in train_phi_category or parts[1] == '' or parts[1] in tmp:
                            continue
                        tmp.add(parts[1])
                        parts[0], parts[1] = parts[0].strip(), parts[1].strip()
                        if parts[0] == "DOCTOR" or parts[0] == "PATIENT":
                            if parts[1] in sentence:
                                start = sentence.index(parts[1]) + len(parts[1])
                                while start < len(sentence) and sentence[start].isalpha():
                                    parts[1] += sentence[start]
                                    start += 1
                                
                    if len(parts) == 2 and parts[0] not in phi_dict:
                        phi_dict[parts[0]] = parts[1]
                    for phi_key, phi_value in phi_dict.items():
                        normalize_time = None
                        if phi_key in normalize_keys:
                            if '=>' in phi_value:
                                temp_phi_values = phi_value.split('=>')
                                phi_value = temp_phi_values[0]
                                normalize_time = temp_phi_values[-1]
                            else:
                                normalize_time = phi_value
                        try:
                            if ' ' in phi_value:
                                matches = [(sentence.index(phi_value), sentence.index(phi_value) + len(phi_value))]
                            else:
                                matches = [(match.start(), match.end()) for match in re.finditer(phi_value, sentence)]
                        except:
                            continue
                        for start, end in matches:
                            if start == end:
                                continue
                            item_dict = {
                                        'phi' : phi_key,
                                        'st_idx' : start + int(boundary),
                                        'ed_idx' : end + int(boundary),
                                        'entity' : phi_value,
                            }
                            if normalize_time is not None:
                                item_dict['normalize_time'] = normalize_time
                            anno_list.append(item_dict)
            return anno_list
        
        seeds = [template.replace("__CONTENT__", data['content']) for data in inputs]
        sep = self.tokenizer.sep_token
        eos = self.tokenizer.eos_token
        pad = self.tokenizer.pad_token
        pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        """Generate text from a trained model."""
        texts = self.tokenizer(seeds, return_tensors = 'pt', padding=True).to(self.device)
        outputs = []
        prediction = []
        with torch.cuda.amp.autocast():
            output_tokens = self.model.generate(**texts, max_new_tokens=50, pad_token_id = pad_idx,
                                            eos_token_id=self.tokenizer.convert_tokens_to_ids(eos))
            preds = self.tokenizer.batch_decode(output_tokens)
            prediction = [i.replace(pad, "").replace(eos, "").strip() for i in preds]
            for idx, pred in enumerate(preds):
                phi_infos = pred.replace(pad, "").replace(eos, "").strip()
                annotations = get_anno_format(inputs[idx]['content'] , phi_infos , inputs[idx]['idx'])

                for annotation in annotations:
                    if 'normalize_time' in annotation:
                        outputs.append(f'{inputs[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}\t{annotation["normalize_time"]}')
                    else:
                        outputs.append(f'{inputs[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}')
        return outputs, prediction
    
    def test(self, dataset):
        print("Evaluating Test Set ...")
        dataset = list(dataset['train'])   
        print(f"Total {len(dataset)} in testing datset!") 
        self.model = self.model.from_pretrained(self.config.model_path).to(self.device)
        self.model.eval()
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        self.model = accelerator.prepare(self.model)
        predictions = []
        with open(f"./{self.config.output_file}",'w',encoding='utf8') as f:
            for i in tqdm(range(0, len(dataset), self.config.batch_size)):
                with torch.no_grad():
                    seeds = dataset[i:i+self.config.batch_size]
                    seeds = [i for i in seeds if i['content'] != None]
                    outputs, prediction = self.aicup_predict(inputs=seeds)
                    predictions.extend(prediction)
                for o in outputs:
                    f.write(o)
                    f.write('\n')
                    print(o)
                    
        # filename = self.config.output_file[:-4] + ".pkl"
        # with open(filename, "wb") as f:
        #     pkl.dump(predictions, f)
                    