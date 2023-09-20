
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd
import numpy  as np
from tqdm.auto import tqdm
from array import array
import torch
import torchvision
import torchvision.transforms as transforms

from transformers import RobertaTokenizer, RobertaModel
import gc
import os


def SpuriousBiasRounds(
        df,
        column_name,
        column_label,
        file_folder,
        train_size = 50,
        slice=10,
        threshold=0.75,
        elimination_num=None,
        m=64,
        embedding = None,
        save = True,
        cur_remove_list_all = []
        ):
    '''
    Params:
        df: the dataframe containing texts and labels
        column_name: name of the column that containing the text
        column_label: name of the column that containing the label
        file_folder: file folder path
        train_size: training set size 
        slice: the size of each slice 
        threshold: the earlystop threshold 
        elmination rounds: the total elmination rounds to perform (if set an value: regardless of threshold)
        m: the prediction value during one iteration 
        embedding: the list with embeddings (defult: roberta-large)
        save: if new embedding calculated, save or not
        cur_remove_list_all: resume from the current remove list
    Return:
        a list of value: the ordered list of instances that were removed
    Usage:
        remove_all_list = SpuriousBiasRounds(
        df,
        "Text",
        "Sentiment",
        file_folder = "spurious_history_contrast/",
        train_size = 50,
        slice=10,
        threshold=0.75,
        elimination_num=None,
        m=64,
        embedding = None,
        save = True
        cur_remove_list_all = cur_remove_list_all
        )
    '''

    module_instance = MainSpuriousModule(df, column_name, column_label, file_folder=file_folder,train_size=train_size, slice=slice, elimination_num=elimination_num, m=m, threshold=threshold, cur_remove_list_all=cur_remove_list_all, embedding=embedding, save=save)
    remove_all_list = module_instance.main_loop()
    return remove_all_list



def GetRobertaEmbeddings(df, column_name, device, emb_path, save=True):

    # direct return the embedding if already saved
    current_directory = os.getcwd()
    # emb_path = 'roberta_em.npz'

    # Check if the file exists in the current directory
    if os.path.isfile(os.path.join(current_directory, emb_path)):
        print(f"The file '{emb_path}' exists in the current directory.")
        with np.load(emb_path) as data:
          loaded_arrays = [data['arr_%d' % i] for i in range(len(data.files))]
          assert len(loaded_arrays) == len(df)
          embedding_roberta = np.array(loaded_arrays)
          x_squeezed = np.squeeze(embedding_roberta, axis=1)
          return x_squeezed

    text_list = df[column_name].to_list()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large').to(device)
    embedding_list_roberta = []

    # Iterate over the texts
    for text in tqdm(text_list):
        #encoded_input = tokenizer(text, return_tensors='pt')
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        encoded_input = encoded_input.to(device)  # Move the input tensors to the device
        with torch.no_grad():
            output = model(**encoded_input)
            embedding_list_roberta.append(output[1].detach().cpu().numpy())  # Move the embeddings to CPU for storage

        embedding_roberta = np.array(embedding_list_roberta)
    if save:
        # emb_path = 'roberta_em.npz'
        np.savez(emb_path, *embedding_roberta)
        print("embedding_saved.")

    return embedding_list_roberta
# dimension: (sample_pool_size, 1024)

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.length = X_train.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        feature_embedding = self.X_train[ind]
        label = self.y_train[ind]
        feature_embedding = torch.FloatTensor(feature_embedding)
        label = torch.tensor(label)
        return feature_embedding, label


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, X_test):
        self.X_test = X_test
        self.length = X_test.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        feature_embedding = self.X_test[ind]
        feature_embedding      = torch.FloatTensor(feature_embedding)
        return feature_embedding


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train(model, dataloader, optimizer, criterion, device):

    model.train()
    tloss, tacc = 0, 0
    #batch_bar   = tqdm(total=len(trainLoader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, (embedding, label) in enumerate(dataloader):

        optimizer.zero_grad()

        embedding      = embedding.to(device)
        label    = label.to(device)

        logits  = model(embedding)

        loss    = criterion(logits, label)

        loss.backward()

        optimizer.step()

        tloss   += loss.item()
        tacc    += torch.sum(torch.argmax(logits, dim= 1) == label).item()/logits.shape[0]

        #batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              #acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        #batch_bar.update()

        ### Release memory
        del embedding, label, logits
        torch.cuda.empty_cache()

    #batch_bar.close()
    # tloss   /= len(trainLoader)
    # tacc    /= len(trainLoader)
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc

def test(model, test_loader, device):

    model.eval() #set to eval
    test_predictions = []
    with torch.inference_mode():
        for i, embedding in enumerate(test_loader):
            embedding = embedding.to(device)
            logits  = model(embedding)
            predicted_label = torch.argmax(logits, dim=1)
            test_predictions.append(predicted_label)
    return test_predictions



class MainSpuriousModule:
    def __init__(self,
                df,
                column_name,
                column_label,
                file_folder,
                train_size = 50,
                slice = 10,
                threshold=0.75,
                elimination_num=400,
                m=64,
                embedding = None,
                save = True,
                cur_remove_list_all = []
                ):
        self.remove_list_all = cur_remove_list_all # return remove_list_all
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = df
        self.file_folder = file_folder
        self.column_name = column_name
        self.column_label = column_label
        self.train_size = train_size
        self.slice = slice
        self.threshold = threshold
        self.elimination_num = elimination_num
        self.m = m
        if embedding is None:
            print("Calculate Embedding")
            self.texts_em = GetRobertaEmbeddings(self.df, self.column_name, self.device, emb_path=file_folder+"roberta_em.npz", save=save)
        else:
            self.texts_em = embedding
        self.input_dim = (self.texts_em).shape[1]  # TODO: check
        print(self.input_dim)
        self.output_dim = self.df[self.column_label].nunique()

        self.stopsign = False
        self.remove_ite = len(self.remove_list_all) / self.slice


    def main_loop(self):
      while(not self.stopsign):
        self.remove_ite += 1
        print("current iteration: " + str(self.remove_ite))
        cur_remove_list_all = self.EachFilterIteration() # calculate new slice
        file_name = self.file_folder + f"spurious_removed_{len(cur_remove_list_all)}.txt"
        with open(file_name, "w") as file:
            # write each item in the list to the file
            for item in cur_remove_list_all:
                file.write(str(item) + "\n")
      return self.remove_list_all

    def EachFilterIteration(self):
        # return a list of index to be eliminated and a flag of whether to continue
        E_i = {} # dictionary for prediction score

        # remove past instances
        arr = self.texts_em
        # Indices to remove
        indices_to_remove = self.remove_list_all
        # Create a mask where only the rows not in the indices to remove are True
        mask = np.ones(len(arr), dtype=bool)
        mask[indices_to_remove] = False
        # Use the mask to get the new array
        texts_em_new = arr[mask]
        assert texts_em_new.shape[0] == (arr.shape[0] - len(indices_to_remove))

        print(texts_em_new.shape)

        # prepare df
        indices_to_drop = self.remove_list_all
        # Drop rows by index
        df_reduce = self.df.drop(indices_to_drop)
        assert df_reduce.shape[0] == (self.df.shape[0] - len(indices_to_drop))
        df_reduce_pass = df_reduce[[self.column_name, self.column_label]]

        num_test_samples = len(df_reduce_pass) - self.train_size
        assert num_test_samples > 0, "num_test_samples must be positive"

        E_i = self.train_test_process(E_i, texts_em_new, df_reduce_pass, num_test_samples, epochNum=10, lr=0.01)

        spurious_df = pd.DataFrame.from_dict(E_i,orient = 'index').rename(columns={0: "total", 1: "right"})
        spurious_df["ratio"] = spurious_df.right / spurious_df.total
        spurious_df = spurious_df.sort_values(by='ratio', ascending=False)
        spurious_df.to_csv("temp_check.csv")



        if not self.elimination_num: # stopsign based on threshold
            print("check: " + str(self.threshold))
            check_df_threshold = spurious_df[spurious_df["ratio"] > self.threshold]
            if len(check_df_threshold) > 0:
                remove_list = check_df_threshold.index.tolist()[:self.slice]
                print("remove_list len: " + str(len(remove_list)))
                self.remove_list_all.extend(remove_list)
                if len(remove_list) < self.slice:
                  self.stopsign = True # do not continue next iteration
            else:
                self.stopsign = True # do not continue next iteration
        else: # stopsign based on elimination num
          remove_list = spurious_df.index.tolist()[:self.slice]
          self.remove_list_all.extend(remove_list)
          if len(self.remove_list_all) >= self.elimination_num:
             self.stopsign = True # do not continue next iteration

        print("current remove list len: " + str(len(self.remove_list_all)))
        return self.remove_list_all


    def train_test_process(self, E_i, texts_em_new, df_reduce_pass, num_test_samples, epochNum=10, lr=0.01):
        for i in tqdm(range(self.m)):
            # collect memory
            torch.cuda.empty_cache()
            collect = gc.collect()

            # split
            X_train, X_test, y_train, y_test = train_test_split(texts_em_new, df_reduce_pass, train_size=self.train_size, test_size=num_test_samples, shuffle=True)
            y_train = np.asarray(y_train[self.column_label])
            TrainData = TrainDataset(X_train, y_train)
            testData = TestDataset(X_test)
            trainLoader = torch.utils.data.DataLoader(TrainData, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            testLoader = torch.utils.data.DataLoader(testData, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
            model = LinearClassifier(self.input_dim, self.output_dim).to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # train
            for epoch in range(epochNum):
                curr_lr                 = float(optimizer.param_groups[0]['lr'])
                train_loss, train_acc   = train(model, trainLoader, optimizer, criterion, self.device)
                if(i % 10 == 0):
                    if epoch == epochNum-1:
                        print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))

            # test
            predictions = test(model, testLoader, self.device)
            results = []

            for batch in predictions:
                for num in batch:
                    results.append(num.cpu().data.numpy())

            y_test['predict'] = results
            y_test['accuracy'] = ((y_test['predict'] == y_test[self.column_label]))
            y_test = y_test.reset_index().rename(columns={"index":"comment_id"})
            for index, row in y_test.iterrows():
                if not row['comment_id'] in E_i:
                    if row['accuracy'] == True:
                        E_i[row['comment_id']] = [1, 1]
                    else:
                        E_i[row['comment_id']] = [1, 0]
                else:
                    E_i[row['comment_id']][0] += 1
                    if row['accuracy'] == True:
                        E_i[row['comment_id']][1] += 1

        return E_i


