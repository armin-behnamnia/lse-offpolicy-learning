

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import time
from tensorboardX import SummaryWriter
writer = None

from model import ModelCifar, ModelV
from data import load_data
from fGAN_eval import evaluate
from loss import CustomLoss, KLLoss, KLLossRev, SupKLLoss
from utils import *
from hyper_params import load_hyper_params
import argparse
import yaml
import numpy as np
from fGAN import VLOSS, QIPSLOSS2

def stable_softmax(logits, dim=-1):
    exp = torch.exp(logits - torch.amax(logits, dim=dim, keepdims=True))
    return exp / torch.sum(exp)

def train(model, criterion, optimizer, scheduler, reader, hyper_params, device):
    
    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([ 0.0 ])
    correct, total = LongTensor([ 0 ]), 0.0
    control_variate = FloatTensor([ 0.0 ])
    ips = FloatTensor([ 0.0 ])
    loss_V = FloatTensor([ 0.0 ])
    loss_Bandit = FloatTensor([ 0.0 ])
    loss_Df = FloatTensor([ 0.0 ])
    v_optimizer, h_optimizer = optimizer
    v_net, h_net = model
    v_criterion, q_ips_criterion, sup_criterion = criterion
    v_scheduler, h_scheduler = scheduler
    s = 0
    for x, y, action, delta, prop in reader.iter():
        s += 1
        
        x, y, action, delta, prop = x.to(device), y.to(device), action.to(device), delta.to(device), prop.to(device)
        bsz = len(x)

        if s % hyper_params.GAN.V_update == 0:
            # Train V
            v_net.train()
            h_net.eval()

            with torch.no_grad():
                h = h_net(x).detach()
                h = stable_softmax(h, dim = 1)
                h_sample = torch.multinomial(h, 1).squeeze(-1)
                # print("h sample =", h_sample)
            for i in range(hyper_params.GAN.V_step):
                v_0 = v_net(x, action)

                v_th = v_net(x, h_sample)
                loss = -v_criterion(v_th, v_0)
                loss.backward()
                # print("V0 =", v_0.mean().item(), ", V_th=", v_th.mean().item(), 'VLOSS=', loss.item())
                # print("V0 =", v_0.max().item(), ", V_th=", v_th.max().item())
                # print("V0 =", v_0.min().item(), ", V_th=", v_th.min().item())
                loss_V += loss.item()

                v_optimizer.step()
                v_optimizer.zero_grad()

        # train h
        v_net.eval()
        h_net.train()

        h = h_net(x)
        with torch.no_grad():
            v_0 = v_net(x, action)

        h_probs = stable_softmax(h, dim = 1)
        # print(v_0.mean().item(), h_probs.mean().item())
        #loss_Q = Q_criterion(v_fake)
        # print(prop, h_probs[torch.arange(0, len(h)), action])
        loss = q_ips_criterion(v_0, prop, h_probs[torch.arange(0, len(h)), action]) * hyper_params.experiment.regularizers.KL
        loss_Df += loss.item()
        # print('DF Loss=', loss.item())
        if hyper_params.experiment.feedback == 'bandit':
            l = sup_criterion(h_probs, action, delta, prop)
            loss += l
            loss_Bandit += l.item()
        elif hyper_params.experiment.feedback is None:
            pass
        else:
            raise ValueError(f'Feedback type {hyper_params.experiment.feedback} is not valid.')        

        # loss_Q.backward()
        # q_optimizer.step()
        # q_optimizer.zero_grad()
        # loss_Q = loss_Q.item()
        
        
        if hyper_params.experiment.regularizers:
            if 'SupKL' in hyper_params.experiment.regularizers:
                loss += SupKLLoss(h, action, delta, prop, hyper_params.experiment.regularizers.eps) * hyper_params.experiment.regularizers.SupKL
        loss.backward()
        h_optimizer.step()
        h_optimizer.zero_grad()

        # Log to tensorboard
        # writer.add_scalar('train loss V', loss_V.item(), total_batches)
        # writer.add_scalar('train loss Q', loss_Q.item(), total_batches)
        # writer.add_scalar('main loss', main_loss, total_batches)

        # Metrics evaluation
        total_loss += loss.item()
        # main_loss += loss_V.item()
        control_variate += torch.mean(h_probs[range(action.size(0)), action] / prop).item()
        ips += torch.mean((delta * h_probs[range(action.size(0)), action]) / prop).item()
        predicted = torch.argmax(h_probs, dim=1)
        # print(predicted, y)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        total_batches += 1.0

    v_scheduler.step()
    h_scheduler.step()
        
    metrics['Bandit Loss'] = round(float(loss_Bandit) / total_batches, 4)
    metrics['Df loss'] = round(float(loss_Df) / total_batches, 4)
    metrics['V loss'] = round(float(loss_V) / total_batches, 4)
    metrics['loss'] = round(float(total_loss) / total_batches, 4)
    metrics['Acc'] = round(100.0 * float(correct) / float(total), 4)
    metrics['CV'] = round(float(control_variate) / total_batches, 4)
    metrics['SNIPS'] = round(float(ips) / float(control_variate), 4)

    return metrics

def main(config_path, device='cuda:0', return_model=False):

    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = load_hyper_params(config_path)
    print(hyper_params)
    if hyper_params.experiment.regularizers:
        if 'KL' in hyper_params.experiment.regularizers:
            print(f"--> Regularizer KL added: {hyper_params.experiment.regularizers.KL}")
        if 'KL2' in hyper_params.experiment.regularizers:
            print(f"--> Regularizer Reverse KL added: {hyper_params.experiment.regularizers.KL2}")
        if 'SupKL' in hyper_params.experiment.regularizers:
            print(f"--> Regularizer Supervised KL added: {hyper_params.experiment.regularizers.SupKL}")

    # Initialize a tensorboard writer
    global writer
    path = hyper_params['tensorboard_path']
    writer = SummaryWriter(path)

    # Train It..
    train_reader, test_reader, val_reader = load_data(hyper_params)

    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(len(test_reader)))

    sup_criterion = CustomLoss(hyper_params)
    v_criterion = VLOSS(divergence=hyper_params.GAN.divergence)
    q_ips_criterion = QIPSLOSS2(divergence=hyper_params.GAN.divergence)
    criterion = (v_criterion, q_ips_criterion, sup_criterion)
    try:
        best_metrics_total = []
        for exp in range(hyper_params.experiment.n_exp):
            h_net = ModelCifar(hyper_params).to(device)
            v_net = ModelV(hyper_params).to(device)
            model = (v_net, h_net)

            v_optimizer = torch.optim.Adam(
                v_net.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
            )
            h_optimizer = torch.optim.SGD(
                h_net.parameters(), lr=hyper_params['lr'], momentum=0.9, weight_decay=hyper_params['weight_decay']
            )
            # v_scheduler = torch.optim.lr_scheduler.StepLR(v_optimizer, step_size=20, gamma=0.5, verbose=True)
            # h_scheduler = torch.optim.lr_scheduler.StepLR(h_optimizer, step_size=20, gamma=0.5, verbose=True)
            v_scheduler = torch.optim.lr_scheduler.OneCycleLR(v_optimizer, max_lr=hyper_params['max_lr'], epochs=hyper_params['epochs'], steps_per_epoch=len(train_reader))
            h_scheduler = torch.optim.lr_scheduler.OneCycleLR(h_optimizer, max_lr=hyper_params['max_lr'], epochs=hyper_params['epochs'], steps_per_epoch=len(train_reader))
            file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")
            file_write(hyper_params['log_file'], f"################################ MODEL ITERATION {exp + 1}:\n--------------------------------")
            best_acc = 0
            best_metrics = None
            optimizer = (v_optimizer, h_optimizer)
            scheduler = (v_scheduler, h_scheduler)
            for epoch in range(1, hyper_params['epochs'] + 1):
                epoch_start_time = time.time()
                
                # Training for one epoch
                metrics = train(model, criterion, optimizer, scheduler, train_reader, hyper_params, device)
                
                string = ""
                for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
                string += ' (TRAIN)'

                for metric in metrics: writer.add_scalar(f'Train_metrics/exp_{exp}/' + metric, metrics[metric], epoch - 1)

                # Calulating the metrics on the validation set
                metrics = evaluate(model, criterion, val_reader, hyper_params, device)
                string2 = ""
                for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
                string2 += ' (VAL)'

                for metric in metrics: writer.add_scalar(f'Validation_metrics/exp_{exp}/' + metric, metrics[metric], epoch - 1)

                ss  = '-' * 89
                ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
                ss += string
                ss += '\n'
                ss += '-' * 89
                ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
                ss += string2
                ss += '\n'
                ss += '-' * 89

                if metrics['Acc'] > best_acc:
                    best_acc = metrics['Acc']

                    metrics = evaluate(model, criterion, test_reader, hyper_params, device)
                    string3 = ""
                    for m in metrics: string3 += " | " + m + ' = ' + str(metrics[m])
                    string3 += ' (TEST)'

                    ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
                    ss += string3
                    ss += '\n'
                    ss += '-' * 89

                    for metric in metrics: writer.add_scalar(f'Test_metrics/exp_{exp}/' + metric, metrics[metric], epoch - 1)
                    best_metrics = metrics

                file_write(hyper_params['log_file'], ss)
            best_metrics_total.append(best_metrics)
            
    except KeyboardInterrupt: print('Exiting from training early')

    writer.close()

    model_summary = {k: [] for k in best_metrics_total[0].keys()}
    for metric in best_metrics_total:
        for k, v in metric.items():
            model_summary[k].append(v)
    for k, v in model_summary.items():
        model_summary[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}

    file_write(hyper_params['summary_file'], yaml.dump(model_summary))

    if return_model == True: return model
    return best_metrics_total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to experiment config file.')
    parser.add_argument('-d', '--device', required=True, help='Device', type=str)
    args = parser.parse_args()
    best_metrics = main(args.config, device=args.device)
    # cr = VLOSS(divergence='RKL')
