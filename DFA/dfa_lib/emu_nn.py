import torch
import numpy as np
import torch.nn.functional as F

from torch import autograd
from torch import Tensor
from torch.autograd.function import once_differentiable

def complex_gaussian(shape):
    """
    make a standard complex iid Gaussian matrix
    """
    A = np.random.normal(size=(shape[0], shape[1], 2)).astype('complex64')
    A[:,:,1] *= 1j
    A = np.sum(A, axis=2)
    A = torch.from_numpy(A)
    A.requires_grad_(False)
    return A


def get_measurements(opu_input: torch.Tensor, TM: torch.Tensor, noise = False, expo=1, snr=10):
    """
    do the random projection to get magnitude measurements
    expo: Coefficient that lowers resulted input. It emulates exposition in the camera and should be taken
    to make max output value equals to 255. It depends on matrix sizes, задаётся в LinearDFAModule::init().
    snr: signal to noise ratio
    """

    opu_input = opu_input.type(torch.complex64)
    measurement = torch.mm(opu_input, TM)

    if noise == True:
        noise = complex_gaussian((opu_input.shape[0], TM.shape[1]))
        noise = noise.to(device)
        """
        шум сделан 'как-то', сейчас snr скорее амплитуда шума, чем соотношение сигнал/шум
        """
        measurement = torch.abs(measurement)**2 * expo + torch.abs(noise)**2 * 255 * expo / snr
        #measurement = torch.abs(measurement + noise / snr)**2 * expo вариант где меняется сама матрица
    else:
        measurement = torch.abs(measurement)**2 * expo

    measurement = torch.round(measurement)
    measurement[measurement <= 0] = 1
    measurement[measurement > 255] = 255

    return measurement


def rand_proj(anchor, opu_input, *args, **kwargs):
    """
    Makes random projection. Check arcticle "Linear Optical Random Projections Without Holography"
    Якорь - опорный вектор, нужен чтобы из квадрата модуля матрицы получить саму матрицу. Задается в LinearDFAError::init()
    """

    #n_samp = opu_input.shape[0]
    anchor_input = anchor - opu_input
    y_anchor = get_measurements(anchor, *args, **kwargs)
    anc_output = get_measurements(anchor_input, *args, **kwargs)
    opu_output = get_measurements(opu_input, *args, **kwargs)

    measurment = (y_anchor + opu_output - anc_output) / (2* torch.sqrt(y_anchor)) / 255 / 255
    #(-1,1) -> (0, 255) so important to divvide by 255^2 ~ 65000
    return measurment


def to_slm(opu_input: torch.Tensor):
    """
    нормирует вектор opu_input и переводит его в int8 (0-255)
    """
    k = 255.0 / torch.max(torch.abs(opu_input))
    return torch.round(opu_input * k)


class ErrorDFA(torch.nn.Module):
    """
    Layer that is used to compute an error by common BP.
    Other DFA layers should contain a reference to the error layer
    """

    def __init__(self, in_features, out_features, bias = None):
        
        super(ErrorDFA, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features   
        self.grad_output = None  # during BP it became an grad_output that used in all other layers or "e" according to the article
        self.random_matrix = None
        
        self.anchor = torch.nn.Parameter(torch.from_numpy(np.random.randint(1,256, (1, out_features))).to(torch.float64))
        self.anchor.requires_grad_(False)
        
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # weight initialization
        
        torch.nn.init.kaiming_uniform_(self.weight)
        # torch.nn.init.constant(self.bias, 1)
        
        
    def forward(self, input): 
        return functional.errorDFA.apply(input, self.weight, self.bias, self)

           
class LinearDFA(torch.nn.Module):
    """
    Simple Fully Connected DFA layer. It differs from common pytorch layers by a custom bacward propogation.
    To learn more see LinearDFA.backward()
    """

    def __init__(self, in_features, out_features, error_layer, bias = None, continue_BP = False):
        
        super(LinearDFA, self).__init__()
        self.in_features = in_features
        self.out_features = error_layer.out_features
        self.error_layer = error_layer
        self.continue_BP = continue_BP
        
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.random_matrix = torch.nn.Parameter(complex_gaussian((self.out_features, out_features)))
        self.random_matrix.requires_grad = False

        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # weight initialization
        
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.random_matrix)
        
        inp = torch.empty((1, self.out_features), dtype=torch.complex64, requires_grad=False)
        inp.fill_(255.0)
        out = torch.abs(torch.mm(inp, self.random_matrix))**2
        self.expo = 255.0/torch.max(out).item()
        # torch.nn.init.constant(self.bias, 1)

    
    def forward(self, input):
        return functional.linearDFA.apply(input, self.weight, self.bias, self) 


class functional():
    
    class errorDFA(autograd.Function):
        """
        The heart of the DFA.
        It has 2 static methods. These are what called during:
        outputs = model_fa(inputs)
        loss.backward()
        """

        @staticmethod
        def forward(ctx, inputs, weight, bias, layer):
            """
            The same as common forward pass
            """
            output = inputs.mm(weight.t()) 

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)


            ctx.intermediate_result = layer
            ctx.save_for_backward(inputs, weight, bias, None)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            The same as common backward pass
            
            Args:
                ctx: (context)is used to store the intermediate results
                grad_output: it is the grad_input from previous layer.

            num of inputs have to be equal to the number of forward outputs + ctx
            num of ctx.saved_variables have to be equal to the number of backward outputs

            Returns:
                how ctx.saved variables have to be changed
            """
            input, weight, bias, _ = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None  # they are how weight and bias should be shanged

            layer = ctx.intermediate_result

            layer.grad_output = grad_output.clone()           
            grad_weight = grad_output.t().mm(input)

            grad_input = input*0 + 1.  # when computing gradient of graph input equals 1 so it is the grad of activation function

            return grad_input, grad_weight, grad_bias, None

        
    class linearDFA(autograd.Function):

        """
        The heart of the DFA.
        It has 2 static methods. These are what called during:
        outputs = model_fa(inputs)
        loss.backward()
        """

        @staticmethod
        def forward(ctx, inputs, weight, bias, layer):
            """
            Args:
                ctx (context) is used to store the intermediate results
                layer - an object torch or DFA module (reference to an object). 
                
            Module has reference to the error layer from which the error is taken
            """
            output = inputs.mm(weight.t()) 

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)


            ctx.intermediate_result = layer
            ctx.save_for_backward(inputs, weight, bias, None)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            Instead of W^t random matrix is used
            Args:
                ctx: (context)is used to store the intermediate results
                grad_output: it is the grad_input from previous layer.

            num of inputs have to be equal to the number of forward outputs + ctx
            num of ctx.saved_variables have to be equal to the number of backward outputs

            Returns:
                how ctx.saved variables have to be changed
            """
            input, weight, bias, _ = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None  # they are how weight and bias should be shanged

            layer = ctx.intermediate_result
            weight_fa = layer.random_matrix
            d_input = 0

            e = layer.error_layer.grad_output.clone() #batch_size x num_classes
            rand_mat = weight_fa
            anch = layer.error_layer.anchor
            d_input = rand_proj(anch, to_slm(e), TM = rand_mat, expo = layer.expo, noise = False, snr = 1000) * grad_output
            grad_weight = (d_input).t().mm(input)  # now grad_output is the gradient of everything that is located between layers          

            grad_input = input*0 + 1.  # when computing gradient of graph input equals 1 so it is the grad of activation function

            if layer.continue_BP == True: # compute grad_input to sent it to the torch.nn.Module

                grad_input = d_input.mm(weight)

            return grad_input, grad_weight, grad_bias, None
