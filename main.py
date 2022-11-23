import torch
import logging
import pickle
import tqdm
import numpy as np
import config
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from util import Lang
from model import BiLstmCrf
from mydata import prepare_data


logging.basicConfig(
    level=logging.INFO,
    filename='training.log',
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s -- %(message)s'
)
lang = Lang()
train, valid = prepare_data(data_dir=config.data_dir,
                            filename=config.train_name,
                            lang=lang,
                            batch_size=config.batch_size,
                            split=True)
models = BiLstmCrf(embedding_dim=config.embedding_dim,
                   hidden_dim=config.hidden_dim,
                   vocab_size=lang.n_words,
                   tag_size=lang.n_tags).to(config.device)
optimizer = torch.optim.RMSprop(models.parameters(), lr=0.01)
with open("lang_dict.pkl", "wb") as file:
    pickle.dump(lang, file, pickle.HIGHEST_PROTOCOL)

global_f1 = 0.0
loss_record = []
PRF = []
for epoch in tqdm.tqdm(range(config.epochs)):
    logging.info("The {} epoch in Training".format(epoch))
    models.train()
    for index, (x, y, size) in enumerate(train):
        optimizer.zero_grad()
        loss = models(x, size, y).abs()
        # prediction_scores = models.decode(x, size)
        loss.backward()
        if index%200 == 0:
            logging.info("epoch:{}------step:{}------loss:{}".format(epoch, index, loss.item()))
            loss_record.append(loss.data.item())
        optimizer.step()

    if epoch % 2 != 0:
        continue

    models.eval()
    logging.info("============= Begin Validation =============\n")
    pred, label = [], []
    with torch.no_grad():
        for x, y, size in valid:
            optimizer.zero_grad()
            models.hidden = models.get_state()
            prediction = models.decode(x, size)
            for li in prediction:
                pred.extend(li)
            for i, le in enumerate(size):
                label.extend(y[i, :le].tolist())

    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    report = classification_report(label, pred)
    logging.info(report)
    PRF.append([precision, recall, f1])
    if f1 > global_f1:
        logging.info("Recall is {}".format(str(recall)))
        logging.info("Precision is {}".format(str(precision)))
        logging.info("F1 is {}".format(str(f1)))
        torch.save(models, str(f1) + config.save_name)

np.savetxt("loss.csv", np.array(loss_record), delimiter=",")
np.savetxt("PRF.csv", np.array(PRF), delimiter=",")


