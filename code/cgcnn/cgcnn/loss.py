import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss_bin = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_bin_w = nn.BCEWithLogitsLoss(
            reduction='none', 
            pos_weight=torch.tensor([4]).to(device)    # 6.94, computed from find_bounds.py --dist
        )  
        self.loss_class = nn.CrossEntropyLoss(reduction='none')
        # self.loss_reg = nn.HuberLoss(reduction='none')
        self.loss_reg = nn.MSELoss(reduction='none')
    

    def forward(self, input, target_dict, mask_dict, task_specs):
        '''
        input: dict of form {
            <taskname>: tensor [B, 7] or [B],
            ...
        }
        target/mask_dict: dict of form {
            <taskname>: tensor[B],
            ...
        }
        '''
        losses = []
        # total_loss = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        loss_dict = dict()  # DEBUGGING: stores loss information for each property
        available_labels = 0    # if 0, every crystal in batch had no labels

        # compute each property loss for all crystals in batch
        for prop, value in input.items():
            task = task_specs[prop]['head']
            targ = target_dict[prop].to(value.device)
            mask = mask_dict[prop].to(value.device)

            num_avail = int(mask.sum().detach().cpu())
            available_labels += num_avail

            # Note: empty labels have default value 0
            # reduction none to mask out unlabeled losses in batch
            if task == 'binary':
                prop_loss = self.loss_bin_w(input[prop], targ)
            elif task == 'multiclass':
                prop_loss = self.loss_class(input[prop], targ.long())
            elif task == 'regression':
                prop_loss = self.loss_reg(input[prop], targ)
            else:
                raise ValueError(f"[Loss] Unknown task: {task}")

            mask_float = mask.float()
            # batch loss for each prop
            batch_loss = (prop_loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)
            batch_loss *= task_specs[prop]['weight']
            
            losses.append(batch_loss)   # batch loss for this prop (accounting for invalid labels)
            loss_dict[prop] = {
                'loss': float(batch_loss.detach().cpu()), # batch loss for each prop
                'num_avail': num_avail,
            }
        
        if len(losses) > 0:
            total_loss = torch.stack(losses).sum()
        else:
            total_loss = sum(pred.sum() * 0.0 for pred in input.values())

        loss_dict['available_labels'] = available_labels
        loss_dict['total_loss'] = float(total_loss.detach().cpu())
        return total_loss, loss_dict