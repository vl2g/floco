import os
from tqdm.auto import tqdm
import json
import torch
from torch.utils.data import Dataset
from evaluate import load
from calc_code_bleu import compute_codebleu
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')


def data_visualisation(code_pth, encodings_pth):
    with open(encodings_pth) as file:
        data = file.read()
    js = json.loads(data)
    image_ids = list(js.keys())
    encodings = list(js.values())
    python_codes=[]
    for id in image_ids:
        cdp = code_pth+str(id)+'.py'
        lines=''
        file = open(cdp, 'r')
        if file.read()[0]=='#':
            file = open(cdp, 'r')
            next(file)
            lines = file.read()
        else:
            file = open(cdp, 'r')
            lines = file.read()
        python_codes.append(lines)
    return image_ids, encodings, python_codes


def CodeT5_tokenize():
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    return tokenizer


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, output, lbl_input_ids, imageids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.output = output
        self.lbl_input_ids = lbl_input_ids
        self.imageids = imageids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return (self.imageids[idx], self.input_ids[idx], self.attention_mask[idx], self.lbl_input_ids[idx], self.output[idx])


def data_loading(train_set, val_set, batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def CodeT5_model():
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    return model


def train(train_loader, val_loader, tokenizer, num_epochs, model, device, optimizer, scheduler, logs_pth, checkpoints_pth):
    best_codebleu=0.0
    best_epoch=1
    exact_match_metric = load("exact_match")
    for epoch in range(num_epochs):
            train_running_loss = 0.0
            model.train()
            print("Epoch: ", epoch)
            bleu_score=0.0
            codebleu_score=0.0
            exact_match=0.0
            for image_id, input_id, attention_mask, label, code in tqdm(train_loader):
                optimizer.zero_grad()
                input_id = input_id.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                # forward pass to get outputs
                outputs = model(input_ids = input_id, attention_mask = attention_mask, labels = label)
                # calculate and backpropagate loss to update parameters
                loss = outputs.loss
                train_running_loss+=loss.item()*input_id.size(0)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Change and print the learning rate after every epoch
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # calculate average loss for the epoch
            train_epoch_loss = train_running_loss/len(train_loader.dataset)
            file=open(logs_pth, 'a')
            file.write("Learning Rate: " + str(lr) + "\n")
            file.write("Train Loss after " + str(epoch+1) + " epoch is : " + str(train_epoch_loss)+ "\n")
            
            bleu_score=0.0
            codebleu_score=0.0
            exact_match=0.0
            model.eval()
            with torch.no_grad():
                for image_id, input_id, attention_mask, label, code in tqdm(val_loader):
                        input_id = input_id.to(device)
                        attention_mask = attention_mask.to(device)
                        code = code.to(device)
                        # Generating codes from the model
                        outputs = model.generate(input_ids = input_id, attention_mask = attention_mask, return_dict_in_generate=True, output_scores=True, max_length=1024)
                        # Decode the output to get the predicted code
                        out = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                        # Decode the label to get the actual code
                        program = tokenizer.batch_decode(code, skip_special_tokens=True)
                        
                        # calculate codebleu, bleu and exact match scores for the predicted codes with respect to the actual codes
                        for i in range(len(program)):
                            codebleu = compute_codebleu([out[i]], [[program[i]]], 'python')[0]
                            bleu = compute_codebleu([out[i]], [[program[i]]], 'python')[1][0]
                            EM = exact_match_metric.compute(predictions=[out[i]], references=[program[i]])['exact_match']
                            bleu_score+=bleu
                            exact_match+=EM
                            codebleu_score+=codebleu
                # calculate the average loss, bleu, exact match and codebleu scores for the validation set
                val_epoch_bleu = bleu_score/len(val_loader.dataset)
                val_epoch_EM = exact_match/len(val_loader.dataset)
                val_epoch_codebleu = codebleu_score/len(val_loader.dataset)

            file.write("Validation Bleu score after " + str(epoch+1) + " epoch is : " + str(val_epoch_bleu)+ "\n")
            file.write("Validation Exact Match score after " + str(epoch+1) + " epoch is : " + str(val_epoch_EM)+ "\n")
            file.write("Validation CodeBleu score after " + str(epoch+1) + " epoch is : " + str(val_epoch_codebleu)+ "\n\n")
            # Save the model with the best validation codebleu score
            if val_epoch_codebleu > best_codebleu:
                best_epoch = epoch+1
                best_codebleu = val_epoch_codebleu
                torch.save(model.state_dict(), os.path.join(checkpoints_pth, '{}.pth'.format(epoch+1)))

    file=open(logs_pth, 'a')
    file.write("\n Best validation loss for " + str(best_epoch) + " epoch is : " + str(best_codebleu)+ "\n")


def run():
    # Path to the train codes
    train_code_pth = ''
    # Path to the train encodings
    train_encodings_pth = ''
    # Path to the validation codes
    val_code_pth = ''
    # Path to the validation encodings  
    val_encodings_pth = ''
    # path to a text file to save the logs
    logs_pth = ''
    # path to save checkpoints
    checkpoints_pth = ''
    # batch size for fine-tuning
    batch_size = 16

    # Load the train and validation data from respective folders 
    train_image_ids, train_encodings, train_codes = data_visualisation(train_code_pth, train_encodings_pth)
    val_image_ids, val_encodings, val_codes = data_visualisation(val_code_pth, val_encodings_pth)
    
    # tokenize the train and validation data with CodeT5 tokenizer
    tokenizer = CodeT5_tokenize()
    tokenizer.add_tokens(['[SEP]', 'PARALLELOGRAM', 'RECTANGLE', 'OVAL', 'DIAMOND'], special_tokens=True)
    train_input = tokenizer(train_encodings, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    val_input = tokenizer(val_encodings, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    with tokenizer.as_target_tokenizer():
        train_labels = tokenizer(train_codes, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        val_labels = tokenizer(val_codes, padding='max_length', truncation=True, return_tensors='pt', max_length=512)

    # Set the labels to -100 for the padding tokens
    train_lbl_input_ids = torch.clone(train_labels['input_ids'])
    val_lbl_input_ids = torch.clone(val_labels['input_ids'])
    for mask in range(train_labels['attention_mask'].shape[0]):
        indices = (train_labels['attention_mask'][mask] == 0).nonzero(as_tuple=True)[0]
        train_lbl_input_ids[mask][indices] = -100

    for mask in range(val_labels['attention_mask'].shape[0]):
        indices = (val_labels['attention_mask'][mask] == 0).nonzero(as_tuple=True)[0]
        val_lbl_input_ids[mask][indices] = -100
    
    # Create the train and validation dataset and dataloaders
    train_set = CustomDataset(train_input['input_ids'], train_input['attention_mask'], train_labels['input_ids'], train_lbl_input_ids, train_image_ids)
    val_set = CustomDataset(val_input['input_ids'], val_input['attention_mask'], val_labels['input_ids'], val_lbl_input_ids, val_image_ids)
    train_loader, val_loader = data_loading(train_set, val_set, batch_size)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load pre-trained CodeT5 model from HuggingFace
    model = CodeT5_model()
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # Define hyperparameters
    num_epochs = 100
    lr = 0.00001
    num_batches = len(train_loader)
    num_training_steps = num_epochs*num_batches
    num_warmup_steps = 2450
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, power=2)
    # Train the model and save the checkpoints and logs
    train(train_loader, val_loader, tokenizer, num_epochs, model, device, optimizer, scheduler, logs_pth, checkpoints_pth)
    

run()
