import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_bin = nn.BCELoss
        self.loss_class = nn.CrossEntropyLoss()
        self.loss_reg = nn.HuberLoss()
        pass

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
        total_loss = torch.tensor([0], requires_grad=True)
        loss_dict = dict()  # DEBUGGING: stores loss information for each property
        available_labels = 0    # if 0, every crystal in batch had no labels

        # compute each property loss for all crystals in batch
        for prop, value in input.items():
            targ = target_dict[prop].to(value.device)
            mask = mask_dict[prop].to(value.device)

            num_avail = int(mask.sum().detach().cpu())
            available_labels += num_avail

            # Note: empty labels have default value 0
            # reduction none to mask out unlabeled losses in batch
            if prop == 'binary':
                prop_loss = self.loss_bin(input[prop], targ, reduction='none')
            elif prop == 'multiclass':
                prop_loss = self.loss_class(input[prop], targ, reduction='none')
            elif prop == 'regression':
                prop_loss = self.loss_reg(input[prop], targ, reduction='none')
            else:
                raise ValueError(f"[Loss] Unknown property: {prop}")

            mask_float = mask.float()
            batch_loss = (prop_loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)
            
            total_loss += batch_loss
            loss_dict[prop] = {
                'loss': float(prop_loss.detach().cpu()),
                'num_avail': num_avail,
            }
        
        loss_dict['available_labels'] = available_labels
        loss_dict['total_loss'] = float(total_loss.detach().cpu())
        return total_loss, loss_dict