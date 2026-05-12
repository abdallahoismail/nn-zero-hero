### FINAL IMPLEMENTATION OF THE VALUE CLASS
import numpy as np


# children, op, and label are optianl params
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0  # stores the gradient of L with respect to this value object
        self._backward = (
            lambda: None
        )  # base case for a leaf node in the graph (a node that is not the result of an operation)
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        # out = self + other

        # this line allows us to do a + 3, or b+1 by wrapping the int in value class
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # a local derivative is defined as the derv of the output of an operation with respect to its inputs.
        # It is local because it only depends on the inputs and outputs of the operation and not on the rest of the graph.
        # It is a property of the operation itself and does not depend on how the operation is used in the graph. It is a building block for backpropagation because it allows us to compute the gradient of the final output with respect to any input by chaining together the local derivatives along the path from the input to the output.
        # local dervitive of out with respect to self and other is 1 since dout/dself = 1 and dout/dother = 1
        def _backward():
            # local derv * global derv
            self.grad += (
                1.0 * out.grad
            )  # out.grad is the derv of the final output of the neuron wrt out
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        # out = self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        # in the expression -> out = self * other
        # local dervitive of out with respect to self and other is dout/dself = other.data and dout/dother = self.data
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):  # allow us to do 2 * a by rearranging it to a * 2
        return self * other

    def expo(self):
        x = self.data
        out = Value(np.exp(x), (self,), "exp")

        # a python closure
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):  # self/other
        return self * other**-1

    def __pow__(self, other):  # self**other
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), "f**{other}")

        def _backward():
            self.grad += other * out.grad * self.data ** (other - 1)

        out._backward = _backward
        return out

    # o is a resuklt of a tanh so when we call o._backward()
    def tanh(self):
        # out = tanh(self)
        # defining the local derivative of tanh with respect to its input self
        # dout/dself = 1 - tanh^2 = 1 - out.data^2
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        # a closure is a record that stores a function together with an environment.
        def _backward():  # local d  * global d
            self.grad += (
                (1 - t**2) * out.grad
            )  # out.grad is chained (backpropped) to z.grad thru the local derv of tanh (1-tanh^2)

        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
