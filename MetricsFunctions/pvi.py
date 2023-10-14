import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, pipeline
from tqdm.auto import tqdm
import gc
import argparse



### function to train the model
def train(device, model, dataloader, optimizer, scheduler, criterion = None):

  model.train()
  tloss, tacc = 0, 0 # Monitoring loss and accuracy
  batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')


  # iterate over batches
  for i, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    # get model predictions for the current batch
    if criterion == None:
      output = model(b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels)

      logits = output.logits
      loss = output.loss
    else:
      logits = model(b_input_ids, b_input_mask)
      loss = criterion(logits, b_labels)

    # backward pass to calculate the gradients
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    # The optimizer dictates the "update rule"--how the parameters are
    # modified based on their gradients, the learning rate, etc.
    optimizer.step()

    # Update the learning rate.
    if criterion == None:
      scheduler.step()

    tloss += loss.item()
    tacc    += torch.sum(torch.argmax(logits, dim= 1) == b_labels).item()/logits.shape[0]

    batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                          acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
    batch_bar.update()

    ### Release memory
    del b_input_ids, b_input_mask, b_labels, logits
    torch.cuda.empty_cache()

  # compute the training loss of the epoch
  batch_bar.close()
  tloss   /= len(dataloader)
  tacc    /= len(dataloader)

  return tloss, tacc



### function for evaluating the model
def evaluate(device, model, dataloader, criterion=None):

  model.eval()
  vloss, vacc = 0, 0 # Monitoring loss and accuracy
  batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')


  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for i, batch in enumerate(dataloader):

    # push the batch to gpu
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    # deactivate autograd
    with torch.no_grad():
      if criterion == None:
        output= model(b_input_ids,
                      token_type_ids=None,
                      attention_mask=b_input_mask,
                      labels=b_labels)

        logits = output.logits
        loss = output.loss
      else:
         # model predictions
        logits = model(b_input_ids, b_input_mask)

        # compute the validation loss between actual and predicted values
        loss = criterion(logits, b_labels)


      vloss   += loss.item()
      vacc    += torch.sum(torch.argmax(logits, dim= 1) == b_labels).item()/logits.shape[0]


    # compute the validation loss of the epoch
    batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                        acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
    batch_bar.update()

    ### Release memory
    del b_input_ids, b_input_mask, b_labels, logits
    torch.cuda.empty_cache()

  batch_bar.close()
  vloss   /= len(dataloader)
  vacc    /= len(dataloader)

  return vloss, vacc



def getDataloaders(args, df, text_name):
    train_text, val_text, train_labels, val_labels = train_test_split(df[text_name], df['tox'],
                                                                        random_state=args.seed,
                                                                        test_size=args.test_ratio,
                                                                        stratify=df['tox'])
        
    max_seq_len = min(512, max(len(text) for text in df[text_name]))
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name, max_length = max_seq_len)

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    # for validation set
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())


    #define a batch size
    batch_size = args.batch_size

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)
    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = SequentialSampler(val_data), batch_size=batch_size)

    return train_dataloader, val_dataloader, tokenizer



def prepareData(args):
    df = pd.read_csv(args.input_data_path, sep='\t')
    df.loc[:, 'text_null'] = ' '
    df.rename(columns={args.text_name: 'text', args.label_name: 'tox'}, inplace=True)
    df['tox'] = df['tox'].apply(lambda x: 0 if x == args.negative_label_name else 1)

    train_dataloader, val_dataloader, tokenizer = getDataloaders(args, df, 'text')
    train_dataloader_null, val_dataloader_null, tokenizer_null = getDataloaders(args, df, 'text_null')

    return (df, train_dataloader, val_dataloader, train_dataloader_null, val_dataloader_null, tokenizer, tokenizer_null)



def saveModels(args, device, train_dataloader, val_dataloader, text_type):
    epochs = args.epochs

    # empty dataset no need to train for many epochs
    if args.task_name == "contrast":
       epochs = 3

    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels = 2, output_attentions = False, output_hidden_states = False)
    # push the model to GPU
    model = model.to(device)

    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = args.lr,
                                  eps = 1e-08
                                  )


    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 500, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # clear memory
    torch.cuda.empty_cache()
    gc.collect()

    val_acc_max = 0.0
    best_num = 0

    for epoch in range(epochs):

        print("\nEpoch {}/{}".format(epoch+1, epochs))

        train_loss, train_acc   = train(device, model, train_dataloader, optimizer, scheduler, criterion=None)
        val_loss, val_acc       = evaluate(device, model, val_dataloader, criterion=None)


        print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, float(optimizer.param_groups[0]['lr'])))
        print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))


        if val_acc_max < val_acc:
          MODEL_SAVE_PATH = ""
          if text_type != "":
             MODEL_SAVE_PATH = args.model_output_dir + args.task_name + f"-pvi-model-{text_type}-{epoch+1}.pt"
          else:
             MODEL_SAVE_PATH = args.model_output_dir + args.task_name + f"-pvi-model-{epoch+1}.pt"

          torch.save({
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc ,
            'optimizer_state_dict': optimizer.state_dict(),
          }, MODEL_SAVE_PATH)

          val_acc_max = val_acc

          print("\nCurrent best model is epoch {} with val_acc_max {}\n".format(epoch+1, val_acc_max))
          best_num = epoch + 1

    return best_num



def generateModels(args, device, train_dataloader, val_dataloader, train_dataloader_null, val_dataloader_null):
    best_num = saveModels(args, device, train_dataloader, val_dataloader, "")
    best_num_null = saveModels(args, device, train_dataloader_null, val_dataloader_null, "null")

    return (best_num, best_num_null)



def getBestModels(args, device, text_type, best_num):
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels = 2, output_attentions = False, output_hidden_states = False).to(device)

    model_name = ""
    if text_type != "":
        model_name = f"{args.task_name}-pvi-model-{text_type}-{best_num}.pt"
    else:
        model_name = f"{args.task_name}-pvi-model-{best_num}.pt"
    
    checkpoint = torch.load(args.model_output_dir + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model



def v_entropy(data, model, tokenizer, input_key, batch_size=32):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, top_k=None, device=0)

    entropies = []
    correct = []
    predicted_labels = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        predictions = classifier(batch[input_key].tolist())
        # print(predictions)

        for i in range(len(batch)):
            prob = next(d for d in predictions[i] if d['label'][-1] == str(batch.iloc[i]['tox']))['score']
            entropies.append(-1 * np.log2(prob))
            if input_key == 'text':
              predicted_label = max(predictions[i], key=lambda x: x['score'])['label']
              predicted_labels.append(predicted_label)
              correct.append(predicted_label == batch.iloc[i]['tox'])

    return entropies, correct, predicted_labels



def calculatePVI(args, df, best_model, best_model_null, tokenizer, tokenizer_null):
    df_temp = df.copy()

    df_temp['H_yx'], df_temp['correct_yx'], df_temp['predicted_label'] = v_entropy(data=df_temp, model=best_model, tokenizer=tokenizer, input_key='text')
    df_temp['H_yb'], _, _ = v_entropy(data=df_temp, model=best_model_null, tokenizer=tokenizer_null, input_key='text_null')
    df_temp['PVI'] = df_temp['H_yb'] - df_temp['H_yx']

    with open(args.output_pvi_path + args.task_name + 'pvi.csv', 'w', encoding = 'utf-8-sig') as f:
      df_temp[['PVI']].to_csv(f, index = False)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="original")
    parser.add_argument("--input_data_path", type=str, default="EvaluationData/IMDB_original.tsv", required=True)
    parser.add_argument("--label_name", type=str, default="Sentiment")
    parser.add_argument("--text_name", type=str, default="Text")
    # only support binary label
    parser.add_argument("--negative_label_name", type=str, default="Negative")
    parser.add_argument("--test_ratio", type=float, help="test size ratio, default ratio is 0.2", default=0.2)
    parser.add_argument("--seed", type=int, default=2018)

    parser.add_argument("--model_name", type=str, default="roberta-large", required=True)
    parser.add_argument("--output_pvi_path", type=str, default="./", required=True)
    
    parser.add_argument("--model_output_dir", type=str, default="model/", required=True)
    parser.add_argument("--epochs", type=int, default=7)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'freeze': 1e-3, 'finetune': 1e-5",
                        default=2e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args



if __name__ == 'main':
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   args = get_args()

   df, train_dataloader, val_dataloader, train_dataloader_null, val_dataloader_null, tokenizer, tokenizer_null = prepareData(args)
   best_num, best_num_null = generateModels(args, device, train_dataloader, val_dataloader, train_dataloader_null, val_dataloader_null)
   best_model, best_model_null = getBestModels(args, device, "", best_num), getBestModels(args, device, "null", best_num_null)
   calculatePVI(args, df, best_model, best_model_null, tokenizer, tokenizer_null)

