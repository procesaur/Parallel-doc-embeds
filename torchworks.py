import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from helpers import average_list

from helpers import probably, flatten


class miniNN(nn.Module):

    def __init__(self, num_feature):
        super(miniNN, self).__init__()
        self.layer_out = nn.Linear(num_feature, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_out(x)
        return x


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def custom_loss_with_accu(pred_batch, batch, model):
    pred = flatten(pred_batch)
    pred = torch.sigmoid(pred)
    weights = model.layer_out.weight
    weights = flatten(weights)
    weights = [x.item() for x in list(weights)]
    weights = average_list(weights)
    x = batch-pred
    x = torch.abs(x)
    loss = torch.mean(x)
    correct = len([y for y in x if y < 0.5])
    size = x.size(dim=0)
    acc = 100*correct/size
    return loss, acc


def get_weights(modelpath):
    n = 5
    if modelpath.endswith('_b'):
        n = 6
    model = miniNN(num_feature=n)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=torch.device(device)))
    model.to(device)
    weights = model.layer_out.weight
    weights = flatten(weights)
    weights = [x.item() for x in list(weights)]
    print(weights)
    return weights


def transform_matrices(dfs, rows_ex, cols_ex):
    inputs = []
    outputs = []
    inputs_temp = []
    outputs_temp = []
    wanted = dfs.keys()
    columns = [x for x in dfs["lemma"].columns if "_" in x]
    n = len(columns)

    for i in range(0, n):
        if columns[i] not in rows_ex:
            info = columns[i].split("_")
            author = info[0]
            novel = info[1]

            able = [x for x in columns if x.split("_")[0] != author or x.split("_")[1] != novel]
            able = [x for x in able if x not in cols_ex]
            for a in able:
                input_vector = []
                output = 0
                if a.split("_")[0] == author:
                    output = 1

                for w in wanted:
                    df = dfs[w]
                    value = df[a][i]
                    input_vector.append(value)

                inputs_temp.append(input_vector)
                outputs_temp.append(output)

    distribution = len([x for x in outputs_temp if x == 1])/len([x for x in outputs_temp if x == 0])
    print(distribution)
    for i in range(0, len(outputs_temp)):
        if outputs_temp[i] == 1:
            inputs.append(inputs_temp[i])
            outputs.append(outputs_temp[i])
        elif probably(distribution):
            inputs.append(inputs_temp[i])
            outputs.append(outputs_temp[i])

    distribution = len([x for x in outputs if x == 1]) / len([x for x in outputs if x == 0])
    print(distribution)

    return inputs, outputs


def test_prob_net(lang, langmodel, rows_ex=None, cols_ex=None, modelname='miniNN'):

    if rows_ex is None:
        rows_ex = []
    if cols_ex is None:
        cols_ex = []

    path = "./data/document_embeds/" + lang
    csvs = [x for x in os.listdir(path) if ".csv" in x]
    wanted = ["pos", "word", "lemma", "masked_2", "masked_3", "bert"]
    csvs = [x for x in csvs if x.split(".")[0] in wanted]
    pd_csvs = {}
    for csv in csvs:
        pd_csvs[csv.split(".")[0]] = pd.read_csv(path + "/" + csv, sep=' ')

    inputs, outputs = transform_matrices(pd_csvs, rows_ex, cols_ex)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = miniNN(num_feature=len(csvs))
    model.load_state_dict(torch.load("./data/document_embeds/" + langmodel + '/' + modelname,
                                     map_location=torch.device(device)))
    model.to(device)

    test_dataset = ClassifierDataset(torch.from_numpy(inputs).float(), torch.from_numpy(outputs).long())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
            sm = nn.Softmax(dim=1)
            probs = sm(y_test_pred)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_prob_list.append(torch.max(probs).item())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    tags = ([])
    return tags, y_prob_list


def train_mini(lang, bert=False, rows_ex=None, cols_ex=None, wanted=None, name="miniNN",
                 epochs=100, batch_size=64, lr=0.01, val_size=0.2):

    if wanted is None:
        wanted = ["pos", "word", "lemma", "masked_2", "masked_3"]
    if bert:
        wanted.append("bert")
        name += "_b"
    if rows_ex is None:
        rows_ex = []
    if cols_ex is None:
        cols_ex = []

    out_path = "./data/document_embeds/" + lang + "/" + name
    path = "./data/document_embeds/" + lang
    csvs = [x for x in os.listdir(path) if ".csv" in x]
    csvs = [x for x in csvs if x.split(".")[0] in wanted]
    pd_csvs = {}
    for csv in csvs:
        pd_csvs[csv.split(".")[0]] = pd.read_csv(path + "/" + csv, sep=' ')

    for df in pd_csvs:
        if pd_csvs[df].columns.tolist()[0] == "Unnamed: 0":
            pd_csvs[df] = pd_csvs[df].set_index("Unnamed: 0")
        if df != "bert":
            pd_csvs[df] = pd_csvs[df].transform(lambda x: [1-y for y in x])
    inputs, outputs = transform_matrices(pd_csvs, rows_ex, cols_ex)

    epochs = round(400000/len(outputs))+150
    print(epochs)
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=val_size, stratify=outputs,
                                                      random_state=1)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    train_dataset = ClassifierDataset(torch.tensor(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = miniNN(len(wanted))

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    accuracy_stats = {
        'train': [],
        "val": [],
    }
    loss_stats = {
        'train': [],
        "val": [],
    }

    print("Begin training.")
    best = 0

    for e in range(1, epochs + 1):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss, train_acc = custom_loss_with_accu(y_train_pred, y_train_batch, model)
            train_loss.backward()

            optimizer.step()

            train_epoch_loss += train_loss
            train_epoch_acc += train_acc

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss, val_acc = custom_loss_with_accu(y_val_pred, y_val_batch, model)
                val_epoch_loss += val_loss
                val_epoch_acc += val_acc

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        tloss = train_epoch_loss / len(train_loader)
        vloss = val_epoch_loss / len(val_loader)
        tacc = train_epoch_acc / len(train_loader)
        vacc = val_epoch_acc / len(val_loader)

        print(f'Epoch {e + 0:03}: |'
              f' Train Loss: {tloss:.5f} |'
              f' Val Loss: {vloss:.5f} |'
              f' Train Acc: {tacc:.3f}|'
              f' Val Acc: {vacc:.3f}' +
              str(model.layer_out.weight))

        score = 1-vloss
        if score > best:
            best = score
            torch.save(model.state_dict(), out_path)

    print(model.layer_out.weight)


