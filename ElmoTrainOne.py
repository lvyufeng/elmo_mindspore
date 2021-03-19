import elmo.utils.clip_grads as clip
import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F
import mindspore.ops as P
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

class ElmoTrainOnestepWithLoss(nn.Cell):
    def __init__(self, network, optimizer, scale_update_cell=None, enable_global_norm=True):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = P.GradOperation(get_by_list=True)
        self.enable_global_norm = enable_global_norm
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.loss_scale_manager = scale_update_cell
        self.grad_reducer = F.identity
    def construct(self, inputs, inputs_back, targets, targets_back):
        weights = self.weights
        loss = self.network(inputs, inputs_back, targets, targets_back)
        grads = self.grad(self.network, weights)(inputs, inputs_back, targets, targets_back)

        # grad reducer on grads
        grads = self.grad_reducer(grads)
        #grads = clip.average_gradients([grads])
        #grads = clip.clip_grads(grads)
        grads = P.clip_by_global_norm(grads)
        train_perplexity = P.Exp()(loss)
        succ = self.optimizer(grads)
        return F.depend(train_perplexity, succ)
