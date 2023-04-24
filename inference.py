from tqdm.auto import tqdm
import json
import torch
from torch.utils.data import Dataset
from evaluate import load
from calc_code_bleu import compute_codebleu
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
    def __init__(self, input_ids, attention_mask, output, imageids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.output = output
        self.imageids = imageids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return (self.imageids[idx], self.input_ids[idx], self.attention_mask[idx], self.output[idx])


def data_loading(test_set, batch_size):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


def CodeT5_model():
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    return model


def writing_results(results_pth, test_loader, tokenizer, model, device):
    bleu_score=0.0
    codebleu_score=0.0
    exact_match=0.0
    for image_id, input_id, attention_mask, code in tqdm(test_loader):
        input_id = input_id.to(device)
        attention_mask = attention_mask.to(device)
        code = code.to(device)
        # Generating the code from the model
        outputs = model.generate(input_ids = input_id, attention_mask = attention_mask, return_dict_in_generate=True, output_scores=True, max_length=1024)
        out = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        program = tokenizer.batch_decode(code, skip_special_tokens=True)
        encod = tokenizer.batch_decode(input_id, skip_special_tokens=True)
        exact_match_metric = load("exact_match")
        # Calculating the metrics for each generated code
        for i in range(len(program)):
            codebleu = compute_codebleu([out[i]], [[program[i]]], 'python')[0]
            bleu = compute_codebleu([out[i]], [[program[i]]], 'python')[1][0]
            EM = exact_match_metric.compute(predictions=[out[i]], references=[program[i]])['exact_match']
            bleu_score+=bleu
            exact_match+=EM
            codebleu_score+=codebleu
        # Writing results to the file
        file=open(results_pth, 'a')
        for i in range(len(image_id)):
            file.write(str(image_id[i])+".png\n")
            file.write("Encoding from tokenizer : \n " + encod[i])
            file.write("\n")
            file.write("Original Python Program : \n " + program[i])
            file.write("\n")
            file.write("Output : \n " + out[i])
            file.write("\n \n ")

    bleu = bleu_score/len(test_loader.dataset)
    EM = exact_match/len(test_loader.dataset)
    codebleu = codebleu_score/len(test_loader.dataset)

    print(bleu, codebleu, EM)


def run():
    # Path to the test codes
    test_code_pth = ''
    # Path to the test encodings
    test_encodings_pth = ''
    # Path to the trained model checkpoints
    trained_model_pth = ''
    # Path to file where the generated codes will be stored
    results_pth = ''
    # Batch size for the test data
    batch_size = 16

    # Loading test data
    test_image_ids, test_encodings, test_codes = data_visualisation(test_code_pth, test_encodings_pth)
    print(len(test_image_ids))
    # Tokenize the test data with CodeT5 tokenizer
    tokenizer = CodeT5_tokenize()
    tokenizer.add_tokens(['[SEP]', 'PARALLELOGRAM', 'RECTANGLE', 'OVAL', 'DIAMOND'], special_tokens=True)
    test_input = tokenizer(test_encodings, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    with tokenizer.as_target_tokenizer():
        test_labels = tokenizer(test_codes, padding='max_length', truncation=True, return_tensors='pt', max_length=512)

    # Create the test dataset and dataloader
    test_set = CustomDataset(test_input['input_ids'], test_input['attention_mask'], test_labels['input_ids'], test_image_ids)
    test_loader = data_loading(test_set, batch_size)
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load pre-trained CodeT5 model from HuggingFace
    model = CodeT5_model()
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Load the fine-tuned model for inference on test data
    model.load_state_dict(torch.load(trained_model_pth, map_location=torch.device('cuda:0')))

    # Generate the results
    writing_results(results_pth, test_loader, tokenizer, model, device)


run()
