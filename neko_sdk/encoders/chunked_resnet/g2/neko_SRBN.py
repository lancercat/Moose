import torch
from torch import nn

# class neko_MSRBN2d(nn.BatchNorm2d):
#     def forward(this, input,mask,stats):
#         return super().forward(input),None,None;

#
# class neko_MSRBN2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True):
#         super(neko_MSRBN2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#
#     def forward(self, input,_,__):
#         self._check_input_dim(input)
#
#         exponential_average_factor = 0.0
#
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         # calculate running estimates
#         if self.training:
#             mean = input.mean([0, 2, 3])
#             # use biased var in train
#             var = input.var([0, 2, 3], unbiased=False)
#             n = input.numel() / input.size(1)
#             with torch.no_grad():
#                 self.running_mean = exponential_average_factor * mean\
#                     + (1 - exponential_average_factor) * self.running_mean
#                 # update running_var with unbiased var
#                 self.running_var = exponential_average_factor * var * n / (n - 1)\
#                     + (1 - exponential_average_factor) * self.running_var
#         else:
#             mean = self.running_mean
#             var = self.running_var
#
#         input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#         if self.affine:
#             input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
#
#         return input,None,None
class neko_MSRBN2d(nn.BatchNorm2d):
    def __init__(this, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(neko_MSRBN2d, this).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def mean_var(this,input, mask):
        if(mask is None):
            inm=input;
            return inm.numel()// input.size(1),inm.mean([0,2,3]),inm.var([0,2,3],unbiased=False)
        else:
            num_dims = len(input.shape[2:])
            _dims = (0,) + tuple(range(-num_dims, 0))
            _slice = (None, ...) + (None,) * num_dims
            num_elements = mask.sum(_dims)
            mean = (input * mask).sum(_dims) / num_elements  # (C,)
            var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)
        return num_elements,mean,var;

    def forward(this, input,mask,stats):

        this._check_input_dim(input)

        exponential_average_factor = 0.0

        if this.training and this.track_running_stats:
            if this.num_batches_tracked is not None:
                this.num_batches_tracked += 1
                if this.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(this.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = this.momentum

        # calculate running estimates
        if this.training:
            n, mean, var = this.mean_var(input, mask);
            if stats is not None:
                stats["means"].append(mean);
                stats["vars"].append(var);
            with torch.no_grad():
                this.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * this.running_mean
                # update running_var with unbiased var
                this.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * this.running_var
        else:
            mean = this.running_mean
            var = this.running_var

        #https://discuss.pytorch.org/t/batchnorm-and-back-propagation/65765/5
        output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + this.eps))
        if this.affine:
            output = output * this.weight[None, :, None, None] + this.bias[None, :, None, None]

        if(mask is not None):
            return output* mask,mask,stats
        else:
            return output,mask,stats
