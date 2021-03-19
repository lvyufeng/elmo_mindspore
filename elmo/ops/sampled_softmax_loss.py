import mindspore
import numpy as np
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import _Loss, _check_label_dtype

class SampledSoftmaxLoss(_Loss):
    r"""
    Computes the sampled softmax training loss.

    Args:
        num_sampled (int): The number of classes to randomly sample per batch.
        num_classes (int): The number of possible classes.
        num_true (int): The number of target classes per training example.
        sampled_values (Tuple):  Tuple of (`sampled_candidates`, `true_expected_count`,
            `sampled_expected_count`) returned by a `*CandidateSampler` function.
            Default to None, `UniformCandidateSampler` is applied.
        remove_accidental_hits (bool): Whether to remove "accidental hits"
            where a sampled class equals one of the target classes.  Default is True.
        seed (int): Random seed for candidate sampling. Default: 0
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **weights** (Tensor) - Tensor of shape (C, dim).
        - **bias** (Tensor) - Tensor of shape (C).  The class biases.
        - **labels** (Tensor) - Tensor of shape (N, num_true), type `int64, int32`. The
          target classes.
        - **inputs** (Tensor) - Tensor of shape (N, dim). The forward activations of
          the input network.

    Outputs:
        Tensor, a tensor of shape (N) with the per-example sampled softmax losses.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> mindspore.set_seed(1)
        >>> loss = nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1)
        >>> weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        >>> biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        >>> labels = Tensor([0, 1, 2])
        >>> inputs = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)
        >>> output = loss(weights, biases, labels, inputs)
        >>> print(output)
        [4.6051701e+01 1.4000047e+01 6.1989022e-06]
    """

    def __init__(self, num_sampled, num_classes, num_true=1,
                 sampled_values=None, remove_accidental_hits=True, seed=0,
                 reduction='none'):
        super(SampledSoftmaxLoss, self).__init__(reduction)

        if num_true < 1:
            raise ValueError(f"num_true {num_true} is less than 1.")
        if seed < 0:
            raise ValueError(f"seed {seed} is less than 0.")
        if num_sampled > num_classes:
            raise ValueError(f"num_sampled {num_sampled} is great than num_classes {num_classes}.")
        if num_true > num_classes:
            raise ValueError(f"num_true {num_true} is great than num_classes {num_classes}.")
        if sampled_values is not None:
            if not isinstance(sampled_values, (list, tuple)):
                raise TypeError(f"sampled_values {sampled_values} is not a list.")
            if len(sampled_values) != 3:
                raise ValueError(f"sampled_values size {len(sampled_values)} is not 3.")

        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits
        self.seed = seed
        self.sampler = P.LogUniformCandidateSampler(
            num_true,
            num_sampled,
            True,
            num_classes,
            seed)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.exp = P.Exp()
        self.log = P.Log()
        self.slice_op = P.Slice()
        self.matmul = P.MatMul(False, True)
        self.gather_v2 = P.Gather()
        self.reduce_max_true = P.ReduceMax(True)
        self.reduce_sum = P.ReduceSum()
        self.reduce_sum_true = P.ReduceSum(True)
        self.concat_dim0 = P.Concat(0)
        self.concat_dim1 = P.Concat(1)
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.mul = P.Mul()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()
        self.compute_accidental_hits = P.ComputeAccidentalHits(num_true)
        self.scatter_nd = P.ScatterNd()
        
    def construct(self, weights, biases, labels, inputs):
        _check_label_dtype(self.dtype(labels), self.cls_name)

        logits, labels = self._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
            num_true=self.num_true,
            sampled_values=self.sampled_values,
            subtract_log_q=True)

        labels = ops.stop_gradient(labels)
        x = self._softmax_cross_entropy(logits, labels)
        return x

    def _softmax_cross_entropy(self, logits, targets):
        stable_exp_logits = self.exp(logits - self.reduce_max_true(logits, 1))
        pred = stable_exp_logits / self.reduce_sum_true(stable_exp_logits, 1)
        return -self.reduce_sum(targets * self.log(pred + 1.0e-20), 1)

    def _compute_sampled_logits(self, weights,
                                biases,
                                labels,
                                inputs,
                                num_true=1,
                                sampled_values=None,
                                subtract_log_q=True):
        """Helper function for SampledSoftmaxLoss functions.

        Computes sampled output training logits and labels suitable

        Note: In the case where num_true > 1, we assign to each target class
        the target probability 1 / num_true so that the target probabilities
        sum to 1 per-example.

        Args:
            weights (Tensor): Tensor of shape `[num_classes, dim]`.
            biases (Tensor): Tensor of shape `[num_classes]`.
            labels (Tensor): Tensor of shape `[batch_size, num_true]`. The target classes.
            inputs (Tensor): Tensor of shape `[batch_size, dim]`.  The forward
                activations of the input network.
            num_true (int): The number of target classes per training example.
            sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `UniformCandidateSampler` function.
            subtract_log_q: A `bool`.  whether to subtract the log expected count of
                the labels in the sample to get the logits of the true labels.
                Default is True.
        Returns:
            out_logits: `Tensor` object with shape
                `[batch_size, num_true + num_sampled]`
            out_labels: A Tensor object with the same shape as `out_logits`.
        """

        if not labels.dtype == mstype.int32:
            labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1, num_true))
        labels_flat = self.reshape(labels, (-1,))

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape is [batch_size, 1] tensor
        #   sampled_expected_count shape is [num_sampled] tensor
        if sampled_values is None:
            labels = self.cast(labels, mstype.int64)
            sampled_values = self.sampler(labels)

        (sampled, true_expected_count, sampled_expected_count) = sampled_values
        sampled = ops.stop_gradient(sampled)
        true_expected_count = ops.stop_gradient(true_expected_count)
        sampled_expected_count = ops.stop_gradient(sampled_expected_count)

        if not sampled.dtype == mstype.int32:
            sampled = self.cast(sampled, mstype.int32)
        all_ids = self.concat_dim0((labels_flat, sampled))
        all_w = self.gather_v2(weights, all_ids, 0)

        n_true = self.shape(labels_flat)[0]
        n_sampled = self.shape(sampled)[0]
        n_dim = self.shape(all_w)[1]

        # true_w shape is [batch_size * num_true, dim]
        true_w = self.slice_op(all_w, [0, 0], [n_true, n_dim])
        sampled_w = self.slice_op(all_w, [n_true, 0], [n_sampled, n_dim])
        sampled_logits = self.matmul(inputs, sampled_w)

        all_b = self.gather_v2(biases, all_ids, 0)
        true_b = self.slice_op(all_b, [0], [n_true])
        sampled_b = self.slice_op(all_b, [n_true], [n_sampled])

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        new_true_w_shape = (-1, num_true, n_dim)
        row_wise_dots = self.mul(self.expand_dims(inputs, 1),
                                 self.reshape(true_w, new_true_w_shape))

        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = self.reshape(row_wise_dots, (-1, n_dim))
        true_logits = self.reshape(self.reduce_sum(dots_as_matrix, 1), (-1, num_true))
        true_b = self.reshape(true_b, (-1, num_true))
        true_logits += true_b
        sampled_logits += sampled_b

        if self.remove_accidental_hits:
            acc_hits = self.compute_accidental_hits(labels, sampled)
            acc_indices, acc_ids, acc_weights = acc_hits
            acc_weights_length = acc_weights.shape[0]
            
            # This is how SparseToDense expects the indices.
            acc_indices_2d = self.reshape(acc_indices[:acc_weights_length], (-1, 1))
            acc_ids_2d_int32 = self.reshape(acc_ids[:acc_weights_length], (-1, 1))
            sparse_indices = self.concat_dim1((acc_indices_2d, acc_ids_2d_int32))
            #sparse_indices = self.cast(sparse_indices, mstype.int32)
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = sampled_logits.shape

            # if self.dtype(sampled_logits) != self.dtype(acc_weights):
            #     acc_weights = self.cast(acc_weights, self.dtype(sampled_logits))
            
            # sampled_logits += self.sparse_to_dense(
            #    sparse_indices,
            #    acc_weights,
            #    sampled_logits_shape)
               
        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= self.log(true_expected_count)
            sampled_logits -= self.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = self.concat_dim1((true_logits, sampled_logits))

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = self.concat_dim1((
            self.ones_like(true_logits) / num_true,
            self.zeros_like(sampled_logits)
        ))
        return out_logits, out_labels
    
    def sparse_to_dense(self, sparse_indices, sparse_values, output_shape):
        output = self.scatter_nd(sparse_indices, sparse_values, output_shape)
        
        return output