import torch
import numpy as np
import torch.nn.functional as F

from torch import autograd
from torch import Tensor
from torch.autograd.function import once_differentiable


class ErrorDFA(torch.nn.Module):
    """
    Layer that is used to compute an error by common BP.
    Other DFA layers should contain a reference to the error layer
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = None):
        
        super(ErrorDFA, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features   
        self.grad_output = None  # during BP it became an grad_output that used in all other layers or "e" according to the article
        
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.random_matrix = torch.nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.random_matrix.requires_grad = False
        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # weight initialization
        
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.random_matrix)
        # torch.nn.init.constant(self.bias, 1)
        
        
    def forward(self, input): 
        return functional.errorDFA.apply(input, self.weight, self.bias, self)

           
class LinearDFA(torch.nn.Module):
    """
    Simple Fully Connected DFA layer. It differs from common pytorch layers by a custom bacward propogation.
    To learn more see LinearDFA.backward()
    """

    def __init__(self, in_features: int, out_features: int, error_layer: ErrorDFA, bias: bool = None, continue_BP: bool = False):
        
        super(LinearDFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_error_features = error_layer.out_features
        self.error_layer = error_layer
        self.continue_BP = continue_BP
        
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.random_matrix = torch.nn.Parameter(torch.Tensor(self.out_error_features, in_features))
        self.random_matrix.requires_grad = False
        # self.random_matrix = torch.empty((self.out_error_features, in_features), dtype=torch.float32, requires_grad=False)

        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # weight initialization
        
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.random_matrix)
        
        
        
        # torch.nn.init.constant(self.bias, 1)

    
    def forward(self, input):
        return functional.linearDFA.apply(input, self.weight, self.bias, self) 


class Conv2dDFA(torch.nn.Module):
    """
        Hasn't finished
    """
    def __init__(self, in_chanels, out_chanels, kernel_size):
        super(Conv2dDFA, self).__init__()
        
        # self.weight = torch.Tensor(out_chanels, in_chanels, kernel_size, kernel_size) #in_chanels/groups
        self.weight = torch.nn.Parameter(torch.Tensor(out_chanels, in_chanels, kernel_size, kernel_size))
        self.weight = self.weight.type(torch.float32)
        torch.nn.init.kaiming_uniform_(self.weight)    
    
    def forward(self, input: Tensor) -> Tensor:
        # print(0)
        return conv2dDFA.apply(input, self.weight)


class functional():

    
    class errorDFA(autograd.Function):
        """
        Similar to the common linear fuction execpt for storing output gradient 
        (if it is a last layer than output gradient == gradient of loss function)
        """

        @staticmethod
        def forward(ctx, inputs: Tensor, weight: Tensor, bias, layer):
            """
            The same as common forward pass. Called during outputs = model_dfa(inputs)
            """
            output = inputs.mm(weight.t()) 

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)


            ctx.intermediate_result = layer
            ctx.save_for_backward(inputs, weight, bias, None)

            return output

        @staticmethod
        def backward(ctx, grad_output: Tensor):
            """
            The same as common backward pass. Called during loss.backward()
            
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

            weight_dfa = layer.random_matrix
            grad_input = grad_output.mm(weight_dfa)  # when computing gradient of graph input equals 1 so it is the grad of activation function
            
            return grad_input, grad_weight, grad_bias, None

        
    class linearDFA(autograd.Function):
        """
        The heart of the DFA.
        """

        @staticmethod
        def forward(ctx, inputs: Tensor, weight: Tensor, bias, layer):
            """
            The same as common forward pass. Called during outputs = model_dfa(inputs)
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
        def backward(ctx, grad_output: Tensor):
            """
            Differs common backward pass. Called during loss.backward()
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
            e = layer.error_layer.grad_output.clone()
            
            grad_weight = grad_output.t().mm(input)# now grad_output is the gradient of everything that is located between layer
                        
            weight_dfa = layer.random_matrix
            grad_input = e.mm(weight_dfa)  # when computing gradient of graph input equals 1 so it is the grad of activation function

            if layer.continue_BP == True: # compute grad_input to sent it to the torch.nn.Module

                grad_input = grad_output.mm(weight)

            return grad_input, grad_weight, grad_bias, None

    
    class conv2dDFA(torch.autograd.Function):
        """
        Hasn't finished
        """
        @staticmethod
        def forward(ctx, inputs, weight, bias = None):

            ctx.save_for_backward(inputs, weight, bias)
            output = F.conv2d(inputs, weight)
            
            return output
        
        
        @staticmethod    
        @once_differentiable    
        def backward(ctx, grad_output):        
            input, weight, _ = ctx.saved_tensors
            
            grad_input, grad_weight = convolution_backward(grad_output, input, weight)
            
            return grad_input, grad_weight, None
    
    
    def convolution_backward(grad_out, inputs, weight):
        """
        Hasn't finished
        """
        grad_weight = F.conv2d(inputs.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
        grad_input = F.conv_transpose2d(grad_out, weight)
        # above from pytorch.org but i think should be (2,3) not (0,1)
        
        return grad_input, grad_weight
        # class functional()