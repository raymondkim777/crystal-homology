import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # losses = []
        losses = dict()
        # total_loss = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        # loss_dict = dict()  # DEBUGGING: stores loss information for each property
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
            
            # losses.append(batch_loss)   # batch loss for this prop (accounting for invalid labels)
            losses[prop] = batch_loss
            # loss_dict[prop] = {
            #     'loss': float(batch_loss.detach().cpu()), # batch loss for each prop
            #     'num_avail': num_avail,
            # }
        
        # if len(losses) > 0:
        #     total_loss = torch.stack(losses).sum()
        # else:
        #     total_loss = sum(pred.sum() * 0.0 for pred in input.values())

        # loss_dict['available_labels'] = available_labels
        # loss_dict['total_loss'] = float(total_loss.detach().cpu())
        # return total_loss, loss_dict
        return losses


class GradNorm(nn.Module):

    def __init__(self, task_specs, alpha=1.5, warmup_steps=1):
        super().__init__()

        self.task_specs = task_specs
        self.task_num = len(task_specs)
        self.alpha = alpha
        self.warmup_steps = warmup_steps

        # w_i(0) = 1
        self.task_weights = nn.Parameter(torch.ones(self.task_num))

        # L_i(0)
        self.register_buffer(
            "initial_losses",
            torch.zeros(self.task_num),
        )
        self.register_buffer(
            "initial_losses_set",
            torch.tensor(False),
        )

        self.register_buffer(
            "initial_loss_sum",
            torch.zeros(self.task_num),
        )

        self.register_buffer(
            "initial_loss_count",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, losses, reference_params):
        """
        Parameters
        ----------
        losses:
            Sequence or 1D tensor containing one scalar loss per task.

        reference_params:
            Parameters W at which GradNorm measures gradient norms.
            This should normally be the final shared encoder layer.

        Returns
        -------
        model_loss:
            Sum_i stop_gradient(w_i) * L_i.
            Backpropagate this through the entire CGCNN.

        gradnorm_loss:
            GradNorm's L_grad.
            Backpropagate this only into task_weights.
        """
        # losses is dict of form losses[prop] = loss_value
        losses = [loss_val for _, loss_val in self.task_specs.items()]
        losses = torch.stack(list(losses))

        if losses.ndim != 1:
            raise ValueError(
                "losses must contain one scalar loss per task"
            )

        if losses.numel() != self.task_num:
            raise ValueError(
                f"Expected {self.task_num} losses, but received {losses.numel()}"
            )

        reference_params = tuple(
            param
            for param in reference_params
            if param.requires_grad
        )

        if not reference_params:
            raise ValueError(
                "reference_params contains no trainable parameters"
            )

        # Store L_i(0) before the first parameter update.
        if not self.initial_losses_set.item():
            with torch.no_grad():
                self.initial_losses.copy_(losses.detach())
                self.initial_losses_set.fill_(True)

                if self.initial_loss_count.item() >= self.warmup_steps:
                    self.initial_losses.copy_(
                        self.initial_loss_sum
                        / self.initial_loss_count
                    )

                    self.initial_losses_set.fill_(True)

            # Train with equal weights during the complete warm-up.
            # GradNorm begins on the next optimizer step.
            model_loss = losses.sum()
            return model_loss, None

        if torch.any(self.initial_losses <= 0):
            raise ValueError(
                "GradNorm requires positive initial losses. "
                "A task may have an empty mask or zero loss."
            )

        # ---------------------------------------------------------
        # 1. Calculate ||grad_W L_i|| at the final shared layer.
        # ---------------------------------------------------------

        weighted_gradient_norms = []

        for task_index, task_loss in enumerate(losses):
            task_grads = torch.autograd.grad(
                outputs=task_loss,
                inputs=reference_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )

            # These gradients are constants for L_grad. GradNorm
            # differentiates L_grad only with respect to w_i.
            task_grads = tuple(
                grad.detach()
                for grad in task_grads
            )

            # G_i = ||grad_W [w_i L_i]||_2
            #
            # Computing the norm this way preserves its dependence
            # on w_i, but not on the CGCNN parameters.
            squared_norm = sum(
                (self.task_weights[task_index] * grad)
                .pow(2)
                .sum()
                for grad in task_grads
            )

            weighted_gradient_norms.append(
                torch.sqrt(squared_norm)
            )

        G_i = torch.stack(weighted_gradient_norms)

        # Mean gradient norm G_bar
        G_bar = G_i.mean()

        # ---------------------------------------------------------
        # 2. Calculate relative inverse training rates.
        # ---------------------------------------------------------

        with torch.no_grad():
            loss_ratios = (
                losses.detach() / self.initial_losses
            )

            r_i = loss_ratios / loss_ratios.mean()

        # The paper explicitly treats the target as constant when
        # differentiating L_grad.
        gradient_targets = (
            G_bar * r_i.pow(self.alpha)
        ).detach()

        # L_grad = Sum_i |G_i - G_bar * r_i^alpha|
        gradnorm_loss = torch.abs(
            G_i - gradient_targets
        ).sum()

        # ---------------------------------------------------------
        # 3. Ordinary weighted multitask loss.
        # ---------------------------------------------------------
        #
        # Detaching task_weights prevents the ordinary task losses
        # from updating them. They must be updated only by L_grad.
        #
        # This loss is backpropagated through the entire CGCNN,
        # including the encoder and all prediction heads.
    
        model_loss = torch.sum(
            self.task_weights.detach() * losses
        )

        return model_loss, gradnorm_loss

    @torch.no_grad()
    def renormalize_weights(self):
        """
        Enforce Sum_i w_i = T after every task-weight update.
        """

        weight_sum = self.task_weights.sum()

        if not torch.isfinite(weight_sum):
            raise RuntimeError(
                "GradNorm task weights became non-finite"
            )

        if weight_sum <= 0:
            raise RuntimeError(
                "GradNorm task weights have a non-positive sum. "
                "Reduce the task-weight learning rate."
            )
        
        self.task_weights.clamp_(min=1e-3)
        self.task_weights.mul_(
            self.task_num / weight_sum
        )