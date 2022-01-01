import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from helpers import average_list

from helpers import probably, flatten


class miniNN(nn.Module):

    def __init__(self, num_feature):
        super(miniNN, self).__init__()
        self.layer_out = nn.Linear(num_feature, 1)
        self.relu = nn.ReLU()

    def forward(self, _, i, p):
        c = torch.zeros(p)
        for x in i:
            t = x[0]
            d = torch.FloatTensor(x[1:])
            r = (self.layer_out(d))
            if r >= c[t]:
                c[t] = r
        return c


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def get_epochs(n):
    epochs = round(8000 / n) + 10
    print(epochs)
    return epochs


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc


def get_class_distribution(obj, classes):
    count_dict = {}
    for i in range(0, len(classes)):
        count_dict[i] = 0
    for i in obj:
        count_dict[i] += 1
    return count_dict


def get_class_weights(classes, train_dataset, y_train):
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(y_train, classes).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    return class_weights[target_list], class_weights


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


def equalize_binary_distribution(inputs, outputs):
    inputs_new = []
    outputs_new = []
    distribution = len([x for x in outputs if x == 1])/len([x for x in outputs if x == 0])
    print(distribution)
    for i in range(0, len(outputs)):
        if outputs[i] == 1:
            inputs_new.append(inputs[i])
            outputs_new.append(outputs[i])
        elif probably(distribution):
            inputs_new.append(inputs[i])
            outputs_new.append(outputs[i])

    distribution = len([x for x in outputs if x == 1]) / len([x for x in outputs if x == 0])
    print(distribution)
    return inputs_new, outputs_new


def transform_matrices(dfs, rows_ex, cols_ex):
    inputs = []
    outputs = []

    wanted = dfs.keys()
    columns = [x for x in dfs["lemma"].columns if "_" in x]
    n = len(columns)

    classes = list(set([x.split("_")[0] for x in columns]))
    idx2class = {i: classes[i] for i in range(0, len(classes))}
    class2idx = {v: k for k, v in idx2class.items()}

    for i in range(0, n):

        if columns[i] not in rows_ex:
            info = columns[i].split("_")
            author = info[0]
            novel = info[1]
            input_vectors = []

            able = [x for x in columns if x.split("_")[0] != author or x.split("_")[1] != novel]
            able = [x for x in able if x not in cols_ex]
            if author in [x.split("_")[0] for x in able]:
                for a in able:
                    input_vector = [class2idx[a.split("_")[0]]]
                    for w in wanted:
                        df = dfs[w]
                        value = df[a][i]
                        if w != "bert":
                            value = 1-value
                        input_vector.append(value)

                    input_vectors.append(input_vector)
                inputs.append(input_vectors)
                outputs.append(class2idx[author])

    idx2class = {i: classes[i] for i in range(0, len(classes))}
    class2idx = {v: k for k, v in idx2class.items()}
    return inputs, outputs, idx2class, class2idx


def train_mini(lang, bert=False, rows_ex=None, cols_ex=None, wanted=None, name="miniNN",
                 epochs=None, batch_size=8, lr=0.01, val_size=0.1):

    # preset default params

    if wanted is None:
        wanted = ["pos", "word", "lemma", "masked_2", "masked_3"]
    if bert:
        wanted.append("bert")
        name += "_b"
    if rows_ex is None:
        rows_ex = []
    if cols_ex is None:
        cols_ex = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = "./data/document_embeds/" + lang + "/" + name

    # load data
    path = "./data/document_embeds/" + lang
    csvs = [x for x in os.listdir(path) if ".csv" in x]
    csvs = [x for x in csvs if x.split(".")[0] in wanted]
    pd_csvs = {}
    for csv in csvs:
        pd_csvs[csv.split(".")[0]] = pd.read_csv(path + "/" + csv, sep=' ')

    # generate inputs and outputs from loaded data
    inputs_true, outputs, idx2class, class2idx = transform_matrices(pd_csvs, rows_ex, cols_ex)
    classes = list(idx2class.keys())
    n_classes = len(classes)
    inputs = [x for x in range(0, len(inputs_true))]
    # calculate optimal number of training epochs
    if epochs is None:
        epochs = get_epochs(len(outputs))

    # generate training, validation and test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(inputs, outputs,
                                                              test_size=0.1,
                                                              random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                      test_size=val_size,
                                                      stratify=y_trainval,
                                                      random_state=1)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    train_dataset = ClassifierDataset(torch.tensor(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # get and remember class distribution
    c_weights_all, c_weights = get_class_weights(classes, train_dataset, y_train)
    weighted_sampler = WeightedRandomSampler(
        weights=c_weights_all,
        num_samples=len(c_weights_all),
        replacement=True
    )

    # create data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # specify model
    model = miniNN(len(wanted))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=c_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # training
    print("Begin training.")
    best = 0

    accuracy_stats = {
        'train': [],
        "val": [],
    }
    loss_stats = {
        'train': [],
        "val": [],
    }

    for e in range(1, epochs + 1):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = []
            for x in X_train_batch:
                y_train_pred.append(model(x, inputs_true[int(x.item())],  n_classes))

            y = torch.stack(y_train_pred)
            train_loss = (criterion(y, y_train_batch))
            train_acc = multi_acc(y, y_train_batch)

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

                y_val_pred = []
                for x in X_val_batch:
                    y_val_pred.append(model(x, inputs_true[int(x.item())], n_classes))

                yy = torch.stack(y_val_pred)
                val_loss = criterion(yy, y_val_batch)
                val_acc = multi_acc(yy, y_val_batch)
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

        score = 1 - vloss
        if score > best:
            best = score
            torch.save(model.state_dict(), out_path)

    print(model.layer_out.weight)


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
