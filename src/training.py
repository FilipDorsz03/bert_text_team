import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from src.constants import MODEL_NAME

# need to change preprocessing
#BERT doesnt want to clearing stopwords or marks
def clean_for_bert(text): # we can keep that
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def get_bert_dataloaders(json_path, batch_size=16, test_split=0.2, model_path=None):

    
    df = pd.read_json(json_path, orient="index")
    
    # preparing texts and labels as it was
    texts = [clean_for_bert(t) for t in df["transcription"].tolist()]
    labels_list = df["label"].tolist()
    
    #tokenizer - built for bert instead of our embeding
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) if model_path is None else AutoTokenizer.from_pretrained(model_path)
    # token
    encoded_inputs = tokenizer(
        texts,
        padding='max_length', #fill with 0 to max length (29/233 CUTED)
        truncation=True, #cut if text too long (problem??)
        max_length=512, #maximum length
        return_tensors='pt' #return as tensor
    )
    
    input_ids = encoded_inputs['input_ids'] #tensor of coded text
    attention_masks = encoded_inputs['attention_mask']
    labels = torch.tensor(labels_list, dtype=torch.long)

    # train/test split as it was
    idx_train, idx_test = train_test_split(
        range(len(labels)), test_size=test_split, random_state=42
    ) #idx_train - list of indexs in training group

    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    
    #train_dataset = TensorDataset(input_ids[idx_train], attention_masks[idx_train], labels[idx_train], texts_train) 
    #test_dataset = TensorDataset(input_ids[idx_test], attention_masks[idx_test], labels[idx_test], texts_test)

        #id,mask,label
    train_dataset = TensorDataset(input_ids[idx_train], attention_masks[idx_train], labels[idx_train]) #matrix of coded text, matrix of masks, matrix of labels (train group)
    test_dataset = TensorDataset(input_ids[idx_test], attention_masks[idx_test], labels[idx_test])

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader #outputing reduced number of parameters

def train_bert_classifier(
    model,
    dataloader_train,
    dataloader_test=None,
    epochs=6, #bylo 4           # BERT needs less epochs (3-5)
    learning_rate=2e-5,   # BERT needs smaller learning_rate
    freeze_until_layer=6
):
    # assert isinstance(model, BERTClassifier) 

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu") #idk what to use in anvil
    print(f"Using device: {device}")
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    
    def get_optimizer_params_with_decreasing_lr(model, base_lr, weight_decay=0.005):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        
        # Decaying learning rate for each of the 12 BERT layers
        lr = base_lr
        decay_rate = 0.9
        
        # 1. The Classifier Head (Highest LR)
        head_params = [p for n, p in param_optimizer if 'fc_final' in n]
        optimizer_grouped_parameters.append({'params': head_params, 'lr': lr, 'weight_decay': weight_decay})
        
        # 2. The BERT Layers (Decaying LR)
        # Iterate backwards from layer 11 to 0
        layers = [*model.bert.encoder.layer]
        layers.reverse()
        
        lr *= decay_rate # Start slightly lower for the top BERT layer
        for layer in layers:
            layer_params = [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)]
            bias_params = [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)]
            
            optimizer_grouped_parameters.append({'params': layer_params, 'lr': lr, 'weight_decay': weight_decay})
            optimizer_grouped_parameters.append({'params': bias_params, 'lr': lr, 'weight_decay': 0.0})
            
            lr *= decay_rate # Reduce LR for the next layer down

        return optimizer_grouped_parameters
    
    
    optimizer = optim.AdamW(get_optimizer_params_with_decreasing_lr(model, learning_rate), weight_decay=0.01)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs * len(dataloader_train))
   


    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
        return module 
    
    def partial_freeze_module(module, freeze_until_layer=freeze_until_layer):
        for name, param in module.named_parameters():
            if "encoder.layer" in name:
                try:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num >= freeze_until_layer: 
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                except:
                    param.requires_grad = False
            elif "pooler" in name or "fc_final" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return module 


    model_loss_history = []
    validation_loss_history = []
    # Training loop
    for epoch in range(epochs):
        total_model_loss = 0.0
        model.train()
        model = partial_freeze_module(model)

        # unpacking 3 elements from batch
        for batch in tqdm.tqdm(dataloader_train, desc=f"Training Epoch {epoch+1}/{epochs}"):
            
            b_input_ids = batch[0].to(device) #inputs ID
            b_input_mask = batch[1].to(device) #NEW mask
            b_labels = batch[2].to(device) #targets

            optimizer.zero_grad()
            
            # ID and mask to forward pass
            outputs, _ = model(b_input_ids, b_input_mask)
            
            model_loss = criterion(outputs, b_labels)
            model_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_model_loss += model_loss.item()

        avg_train_loss = total_model_loss / len(dataloader_train)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        model_loss_history.append(avg_train_loss)

        # validation every epoch
        if dataloader_test is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for v_batch in dataloader_test:
                    v_ids = v_batch[0].to(device)
                    v_mask = v_batch[1].to(device)
                    v_labels = v_batch[2].to(device)

                    v_outputs, _ = model(v_ids, v_mask)
                    v_loss = criterion(v_outputs, v_labels)
                    total_val_loss += v_loss.item()

            avg_val_loss = total_val_loss / len(dataloader_test)
            validation_loss_history.append(avg_val_loss)
            print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")


    if dataloader_test is not None:
        plot_loss_curves(model_loss_history, validation_loss_history)

    return model


def plot_loss_curves(model_loss_history, validation_loss_history=None):
    """
    Plots the loss curves on a single plot for direct comparison.
    """
    if validation_loss_history is None:
        validation_loss_history = []

    iterations = range(len(model_loss_history))
    val_iterations = range(len(validation_loss_history))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Model Loss
    ax.plot(iterations, model_loss_history, marker='o', color='blue', label='Model Loss')
    
    # Plot Validation Loss
    if validation_loss_history:
        ax.plot(val_iterations, validation_loss_history, marker='o', color='orange', label='Validation Loss')

    # Formatting
    ax.set_title('Training and Validation Loss Curves')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    import time
    fig.savefig(f"fig_{int(time.time())}.png")
    plt.show()

    return fig