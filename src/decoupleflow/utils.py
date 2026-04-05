import torch
import torch.nn as nn
import torch.optim as optim
import threading


class Optimizer:
    """Thin optimizer wrapper supporting built-in or custom optimizer configs."""

    def __init__(self, model_parameters, optimizer_function, optimizer_parameters):
        if isinstance(optimizer_function, str):
            if optimizer_parameters:
                self.optimizer = LARS(model_parameters, **optimizer_parameters)
            else:
                self.optimizer = LARS(model_parameters, lr=1e-3)
        else:
            if optimizer_parameters:
                self.optimizer = optimizer_function(model_parameters, **optimizer_parameters)
            else:
                self.optimizer = optimizer_function(model_parameters, lr=1e-3)
    
    def zero_grad(self):
        """Clear gradients of wrapped optimizer."""
        self.optimizer.zero_grad()
        
    def get_learning_rate(self):
        """Return current learning rate.

        Returns:
            float: Learning rate of the first parameter group.
        """
        return self.optimizer.param_groups[0]['lr']


class LR_Scheduler:
    """Small wrapper to unify scheduler construction and stepping."""

    def __init__(self, opt, scheduler_fn, scheduler_param):
        self.scheduler = scheduler_fn(opt, **scheduler_param)
        
    def step(self, *arg):
        """Advance wrapped scheduler.

        Args:
            *arg: Optional scheduler step arguments.
        """
        self.scheduler.step(*arg)


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer."""

    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform one optimization step using LARS adaptation.

        Returns:
            None
        """
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
                
    @staticmethod
    def exclude_bias_and_norm(parameter):
        """Check whether a parameter is bias/norm style.

        Args:
            parameter: Model parameter tensor.

        Returns:
            bool: `True` when tensor is 1D.
        """
        return parameter.ndim == 1


class CPUThread(threading.Thread):
    """Thread wrapper that stores function result for later retrieval."""

    def __init__(self, target=None, args=(), **kwargs):
        super(CPUThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self.__result__ = None
        self.__exception__ = None

    def run(self):
        """Execute target function and store result safely."""
        if self._target == None:
            return
        try:
            self.__result__ = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.__result__ = None
            self.__exception__ = e
    
    def get_result(self):
        """Wait for thread completion and fetch result.

        Returns:
            Any: Stored function result.

        Raises:
            ValueError: If thread execution failed or returned no result.
        """
        self.join()
        if self.__exception__ is not None:
            raise ValueError(
                "Here are some error in loss backward, please check your model structure"
            ) from self.__exception__
        else:
            return self.__result__




