
import json
import fire
import pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.utils import JamoTokenizer
from model.data import Corpus
from model.net import CharCNN
from tensorboardX import SummaryWriter

from gluonnlp.data import PadSequence



def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(dataloader), desc='steps', total=len(dataloader)):
        x_bm, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = loss_fn(model(x_bm), y_mb)

        avg_loss += mb_loss.item()
    else: # if 없이 그냥 else만 존재?
        avg_loss /= (step+1)



def main(cfgpath):
    # parsing json
    proj_dir = Path('.')
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')


    # Common params
    tokenizer = JamoTokenizer()


    # model params
    num_classes = params['model'].get('num_classes')
    embedding_dim = params['model'].get('embedding_dim')


    # dataset, dataloader params
    length = params['padder'].get('length')
    batch_size = params['training'].get('batch_size')
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')


    # Creating Model
    model = CharCNN(num_classes=num_classes,
                    embedding_dim=embedding_dim,
                    dic=tokenizer.token2idx)


    # Creating dataset, dataloader
    padder = PadSequence(length)
    tr_ds = Corpus(tr_filepath, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_ds = Corpus(val_filepath, tokenizer, padder)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)


    # Training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    writer = SummaryWriter(log_dir='./runs/exp')

    for epoch in tqdm(range(epochs), desc='epochs'):
        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = loss_fn(model(x_mb), y_mb)
            mb_loss.backward()
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % 300 == 0:
                val_loss = evaluate(model, val_dl, loss_fn, device)
                writer.add_scalars('loss', {'train': tr_loss / (step+1),
                                            'validation': val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch+1, tr_loss, val_loss))

    ckpt = {'model_state_dict' : model.state_dict(),
            'opt_state_dict' : opt.state_dict()}

    savepath = proj_dir / params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)



if __name__ == '__main__':
    fire.Fire(main)



























# -------------------------
#
# -------------------------


