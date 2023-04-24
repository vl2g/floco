import os
from tqdm.auto import tqdm
import random
import json
import torch
from torch.utils.data import Dataset
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')


def data_loading(codes_pth, encodings_pth):
    with open(encodings_pth) as file:
        data = file.read()
    js = json.loads(data)
    encodings = list(js.values())
    python_codes=[]
    for code in os.listdir(codes_pth):
        cdp = codes_pth+code
        lines=''
        file = open(cdp, 'r')
        lines = file.read()
        python_codes.append(lines)
    inputs = encodings
    inputs.extend(python_codes)
    random.shuffle(inputs)
    return inputs


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])
    def __len__(self):
        return self.input_ids.shape[0]


def tokenize(inputs):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    tokenizer.add_tokens(['[SEP]', 'PARALLELOGRAM', 'RECTANGLE', 'OVAL', 'DIAMOND'], special_tokens=True)
    tokenized_inputs = tokenizer(inputs, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    return tokenizer, tokenized_inputs


def masking(tokenized_inputs):
    # create a copy of input_ids tensor to be used as labels for MLM
    tokenized_inputs['labels'] = tokenized_inputs.input_ids.detach().clone()
    # create a random tensor of same shape as input_ids
    rand = torch.rand(tokenized_inputs.input_ids.shape)
    # mask 15% of the tokens in each sequence, while ignoring [CLS], [PAD] and [SEP] tokens 
    mask_arr = (rand < 0.15) * (tokenized_inputs.input_ids != 0) * (tokenized_inputs.input_ids != 1) * (tokenized_inputs.input_ids != 2) * (tokenized_inputs.input_ids != 32100)

    # replace selected tokens with [MASK] token
    selection = []
    for i in range(tokenized_inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
    
    for i in range(tokenized_inputs.input_ids.shape[0]):
        tokenized_inputs.input_ids[i, selection[i]] = 4

    return tokenized_inputs


def MLM_model():
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    return model


def train(epochs, train_loader, model, optim, scheduler, device, logs_pth, checkpoints_pth):
    best_loss=10000
    best_epoch=1
    for epoch in range(epochs):
        train_running_loss = 0.0
        model.train()
        for input_ids, attention_mask, labels in tqdm(train_loader):
            optim.zero_grad()

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # extract loss
            loss = outputs.loss

            train_running_loss+=loss.item()*input_ids.size(0)
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            scheduler.step()

        train_epoch_loss = train_running_loss/len(train_loader.dataset)
        file = open(logs_pth, 'a')
        file.write("Train Loss after " + str(epoch+1) + " epoch is : " + str(train_epoch_loss)+ "\n")

        if train_epoch_loss < best_loss:
            best_epoch = epoch+1
            best_loss = train_epoch_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_pth, '{}.pth'.format(epoch+1)))

    print("Best epoch is ", best_epoch, " with loss ", best_loss)


def run():
    # path to augmented codes
    train_codes_pth = ''
    # path to encodings of FloCo train set
    train_encodings_pth = ''
    # path to a text file to save the logs
    logs_pth = '' 
    # path to save checkpoints
    checkpoints_pth = ''

    # load and shuffle the train data
    train_inputs = data_loading(train_codes_pth, train_encodings_pth)
    print("Pre-training with ", len(train_inputs) ," samples")

    # tokenize the train data with CodeT5 tokenizer
    tokenizer, tokenized_train_inputs = tokenize(train_inputs)
    
    # masking tokens randomly at a probablity of 15%
    masked_train_inputs = masking(tokenized_train_inputs)
    
    train_set = CustomDataset(masked_train_inputs['input_ids'], masked_train_inputs['attention_mask'], masked_train_inputs['labels'])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    
    # Load pre-trained CodeT5 model from HuggingFace
    model = MLM_model()
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # Define hyperparameters
    lr = 0.00001
    epochs = 100
    num_batches = len(train_loader)
    num_warmup_steps = 1100
    num_training_steps = epochs*num_batches
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_polynomial_decay_schedule_with_warmup(optim, num_warmup_steps, num_training_steps, power=2)

    # Train the model and save the checkpoints and logs
    train(epochs, train_loader, model, optim, scheduler, device, logs_pth, checkpoints_pth)
    

run()
