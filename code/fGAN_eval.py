

import torch
import torch.nn.functional as F
writer = None

from loss import SupKLLoss
from utils import *

def evaluate(model, criterion, reader, hyper_params, device):
    
    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([ 0.0 ])
    correct, total = LongTensor([ 0 ]), 0.0
    control_variate = FloatTensor([ 0.0 ])
    ips = FloatTensor([ 0.0 ])
    loss_V = FloatTensor([ 0.0 ])
    loss_Bandit = FloatTensor([ 0.0 ])
    loss_Df = FloatTensor([ 0.0 ])
    v_net, h_net = model
    v_criterion, q_ips_criterion, sup_criterion = criterion
    for x, y, action, delta, prop in reader.iter():
        v_net.eval()
        h_net.eval()
        
        
        x, y, action, delta, prop = x.to(device), y.to(device), action.to(device), delta.to(device), prop.to(device)
        bsz = len(x)

        # Eval V


        with torch.no_grad():
            v_0 = v_net(x, action)
            h = h_net(x)
            h = F.softmax(h, dim = 1)
            h_sample = torch.multinomial(h, 1).squeeze(-1)
            v_th = v_net(x, h_sample)
            loss = -v_criterion(v_0, v_th)
        loss_V += loss.item()


        # Eval h

        with torch.no_grad():
            h = h_net(x)
            v_0 = v_net(x, action)
            h_probs = F.softmax(h, dim = 1)
            loss = -q_ips_criterion(v_0, prop, h_probs[torch.arange(0, len(h)), action]) * hyper_params.experiment.regularizers.KL
        loss_Df += loss.item()
        if hyper_params.experiment.feedback == 'bandit':
            l = sup_criterion(h_probs, action, delta, prop)
            loss += l
            loss_Bandit += l.item()
        elif hyper_params.experiment.feedback is None:
            pass
        else:
            raise ValueError(f'Feedback type {hyper_params.experiment.feedback} is not valid.')        

            
        if hyper_params.experiment.regularizers:
            if 'SupKL' in hyper_params.experiment.regularizers:
                loss += SupKLLoss(h, action, delta, prop, hyper_params.experiment.regularizers.eps) * hyper_params.experiment.regularizers.SupKL

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
        
    metrics['Bandit Loss'] = round(float(loss_Bandit) / total_batches, 4)
    metrics['Df loss'] = round(float(loss_Df) / total_batches, 4)
    metrics['V loss'] = round(float(loss_V) / total_batches, 4)
    metrics['loss'] = round(float(total_loss) / total_batches, 4)
    metrics['Acc'] = round(100.0 * float(correct) / float(total), 4)
    metrics['CV'] = round(float(control_variate) / total_batches, 4)
    metrics['SNIPS'] = round(float(ips) / float(control_variate), 4)

    return metrics