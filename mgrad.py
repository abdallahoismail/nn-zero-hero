import numpy as np


class Value:
    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.grad = 0.0
        self.op = op
        self.label = label
        self.children = set(children)
        self._backward = lambda: None  # empty function

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance (other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), +, out.label)
        
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward 
        return out
    
    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), *, out.label)
        def _backward():
        # self.grad = the derv of the loss wrt to self = local derv * global derv
        # self.grad = dout/dself * out.grad
            self.grad  = other.data * out.grad
            other.grad = self.data * out.grad
            
        out._backward = _backward
        return out
