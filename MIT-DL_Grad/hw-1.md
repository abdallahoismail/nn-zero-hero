Homework 1

6.S898 Deep Learning

Fall 2024

Instructions: There are a total of 29 points for this homework. Each question is marked
with the number of points it’s worth. Some questions are not graded, including all bonus
questions. 

You may either hand-draw or computer-generate plots as you find appropriate,
but just make sure the important trends are clear.

Notation: We will use this math notation from the course webpage. For example, c is a
scalar, b is a vector and W is a matrix. You are encouraged (though not forced) to follow
this notation in a typeset submission, or to the best of your ability in a handwritten
response—bolding vectors may be difficult :).

Math primer: A few questions use terms like “convexity”, “discontinuity” and “differentiability”. To solve the problems, only an informal understanding of these concepts is needed. 

For example, ReLU(x) is continuous because it can be drawn without lifting your pen from the paper. ReLU(x) is not differentiable everywhere since it has a “kink” at x = 0.

##################################################################

# Q1
Approximation (14pt)

For this section, we consider functions represented by ReLU networks with a single real-valued input and a single real-valued output, unless otherwise specified. Let l denote the number of layers. 

For example, a network with l = 2 layers is written:

f(x;W1,W2, b1, b2) = W2 ReLU(W1x + b1) + b2,

where input x and output f(x;W1,W2, b1, b2) are scalars, unless otherwise specified (and l is the number of weight matrices).

1. (1pt) Consider a two-layer ReLU network with width k (i.e., k hidden neurons) with weight matrices W1, W2 and bias vectors b1, b2. W1 is a matrix in R
k×1—write down the shapes of W2, b1 and b2.

`- shape of W matrix (out_dim, in_dim)
## SOLUTION
var      shape
x        scalar, but we can treat as matrix of shape (1,1)
W1       (k, 1) col vector, given in problem
W1x will gives a shape of (k,1)
in linear alegbera to add 2 vectors, they must have the same shape
so the shape of the rxpression w1x+b (i.e. z) is (k, 1) and b1 also (k,1)
b1       (k, 1)  
z1       (k, 1)
out1     (k, 1) out = relu(z)
w2       (1, k)
W2*out1  (1, 1)
b2       (1, 1)
z2       (1, 1)

##################################################################
##################################################################

Problem: 3-Layer Network Dimensions
Consider a 3-layer neural network (two hidden layers, one output layer).

The input $x$ is a feature vector of dimension $d_{in}$.
The first hidden layer has $k_1$ neurons.
The second hidden layer has $k_2$ neurons.
The final output of the network is a vector $y$ of dimension $d_{out}$.

The network is defined by the following equation:$y = W_3 \text{ReLU}(W_2 \text{ReLU}(W_1 x + b_1) + b_2) + b_3$Assuming the input $x$ is represented as a column vector (shape $d_{in} \times 1$), what are the shapes of the weight matrices ($W_1, W_2, W_3$) and the bias vectors ($b_1, b_2, b_3$)?To tackle this, let's start from the inside out, just like last time. The innermost operation is multiplying our input vector $x$ by the first weight matrix $W_1$.If $x$ is a column vector of shape $d_{in} \times 1$, and the first hidden layer needs to output a vector for its $k_1$ neurons (meaning the result of $W_1 x + b_1$ must be $k_1 \times 1$), what must the shape of $W_1$ be?

f(x) = W3 relu(W2 relu(W1X + b1) + b2) + b3
f(x) = W3 relu(W2 relu(z1) + b2) + b3
f(x) = W3 relu(W2 h1 + b2) + b3
f(x) = W3 relu(z2) + b3
f(x) = W3 h2 + b3 # output layer

f(x) = W3 relu(z2) + b3
f(x) = W3 h2 + b3

var        shape
x         (d_in,  1)
W1        (k1, d_in)
W1X       (k1, 1)
b1        (k1, 1)
z1        (k1, 1)
h1        (k1, 1)
W2        (k2, k1)
W2h1      (k2, 1)
b2        (k2, 1)
z2        (k2, 1)
h2        (k2, 1)
W3        (d-out, k2)
W3h2      (d-out, 1)
b3        (d-out, 1)
y=z3      (d-out, 1)

Every weight matrix has shape (out_dim, in_dim), and every bias is (out_dim, 1). out_dim = number of neurons in the layer, and in_dim is the number of inputs coming in from the previous layer
Every weight matrix has shape (out_dim, in_dim), and every bias is (out_dim, 1).

##################################################################
##################################################################

2. (1pt) Write out the expression for a ReLU network with l = 3 layers.
f(x;W1,W2, W3, b1, b2, b3) = W3 @ ReLU(W2 ReLU(W1x + b1) + b2) + b3

##################################################################

3. (1pt) Answer yes or no: For a ReLU network with l ≥ 2 layers, in general, is the network output convex with respect to the input x? Is it concave?

Solution-1:
            f(x;W1,W2, b1, b2) = W2 ReLU(W1x + b1) + b2

z = w1x + b1

out = ReLU(z)

fx = W2 * out + b

df/dx = df/dout * dout/dz * dz/dx =  W2 * 1 * W1 = W1 * W2
f'' = 0, so we cannot determine

solution-2:
f(z) = ReLU(z) -> ReLU function is convex (concave up)

fx = W2 * f(z) + b
a convex function * + number is convex
a convex function * - number is concave --> think of fx = x**2 and -fx = -x**2
we know that W2 is a matrix of real numbers (+, -, 0)
so when we * a convex function (like ReLU) with + and - numbers we get bumpy landscape that contains both convex and concave functions.
Convex: (yes/no) --> No

Concave: (yes/no) --> No

### let's run through sigmoid and tanh as well 
 f(x;W1,W2, b1, b2) = W2 sigmoid(W1x + b1) + b2
- is sigmoid convex or concave? both depending on x, in genral non-convex
- a non-convex function * any real number + or - is still non-convex (different from relu)

f(x;W1,W2, b1, b2) = W2 tanh(W1x + b1) + b2
- is tanh convex or concave? both depending on x, in genral non-convex
- a non-convex function * any real number + or - is still non-convex (different from relu)

bottom line: loss functions (y - f(x)) are non-convex thats why we have local minima problems - even if activation fns are convex, the Weights can change that

##################################################################
##################################################################


4. (5pt) Consider a ReLU network with l layers, each of width k.
(a) (1pt) How many discontinuities can the output have with respect to the input?

zero

the relu function is continous (but not diff at x = 0), and then we add and muliply + or - nums, which does not change the continuity of the function that the NN learns




(b) (1pt) Choose one. 
As a function of the input, the output is always:
(A) linear, 
### (B) piecewise-linear,
(C) polynomial.


(c) (2pt) In general, is the function differentiable at every input? If yes, why? If no:
i. What’s the smallest number of input points at which the function can be
non-differentiable?
# solution: 
the relu function is non-diff at x=0, but let's assume that the function that the NN learned: f(x) = W2 relu(W1x + b1) + b2, if W2 is a matrix of zeros then the function will become a flat line at y=0, which would be diff at all points. so the answer is zero.

ii. For l = 2 layers, what’s the largest number of input points at which the
function can be non-differentiable?

f(x;W1,W2, b1, b2) = W2 ReLU(W1x + b1) + b2
each layer has k neurons, each neuron has a relu with one kink, so the max kinks is k

iii. For a general number of layers l, what’s the largest number of input points
at which the function can be non-differentiable? Choose one:
(A) Constant in l 
(B) Linear in l 
(C) polynomial in l 
### (D) exponential in l.

Hint:
• It is hard to derive an exact answer, so focus on asymptotics.
• How are non-differentiable points related to linear regions?
• Each neuron is a separating hyperplane of the previous layer output.
• Assume that each linear region of the previous layer is divided (into
two) by some hyperplane in the current layer. How does adding a layer
affect the number of linear regions?

# solution
We need to figure out what happens as we add more and more layers ($l$).The hint at the bottom of your problem is the key to unlocking this. It asks: "Assume that each linear region of the previous layer is divided (into two) by some hyperplane in the current layer. 

How does adding a layer affect the number of linear regions?"Let's visualize this.Your first layer creates $k$ corners, which chops your graph into pieces (linear regions).The next layer looks at that already-chopped-up, bumpy graph and applies its own ReLU activations. 

Because it is acting on an already folded input, it can fold each of the existing regions again.Think about repeatedly folding a piece of paper 📄. If you fold it once, you get 2 regions. Fold it again, you get 4. Fold it again, you get 8. The number of creases multiplies with every single fold.Similarly, in our network, every time we add a new layer, it essentially multiplies the number of existing regions by another factor of $k$. After $l$ layers, the number of regions (and therefore the number of corners) grows to roughly $k \times k \times k \dots$ ($l$ times), which is mathematically written as $k^l$.

this is why deeper nets are more effective at approximating any function compaerd to wider nets

in a wider net the number of kinks grows linearly O(n)
in a deep net the number of kinks grows exponentially, k**l, which means that the network can create more complex decison boundaries


iv. Recall from lecture that a 2-layer network (l = 2) with sufficient width k is a universal approximator. Based on your answer to the previous questions,
can deeper ReLU networks (l > 2) be more efficient (in terms of total number
of neurons / hidden units) in approximating some functions?

yes, because the number of kinks grows expoenentailly, k**l

(d) (1pt) In general, is the function differentiable at every input if a tanh nonlinearity is used instead of ReLU? If yes, why? If no, re-do the sub-points of part (c).

f(x;W1,W2, b1, b2) = W2 tanh(W1x + b1) + b2
tanh is diff for all x,
W2 + or - or 0, fx will remain diff


##################################################################
##################################################################
##################################################################


5. (2pt) For a 2-layer ReLU network with width 2 and no biases (i.e., b1 and b2 are all zeros), we aim to find W1 and W2 so that the corresponding network has different smoothness properties. 

For each case,
• If there exist W1 and W2 such that the corresponding property holds for input x ∈ [−5, 5], provide an example of such W1 and W2 and, for that example, provide a plot of the function over x ∈ [−5, 5].
• If there do not exist such W1 and W2, explain why.

# these are the cases to consider
(a) (1pt) The function is linear.
the learned network function is linear at W1 (col) = [1, -1], 
W2 (row) = [1, -1] - they result in a linear function f(x) = x

(b) (1pt) The function has 2 non-differentiable points.
impossible to find W1 and W2 since we do not have any biases, the relu functions' kinks overlap at x=0

(c) (BONUS; 0pt) The function is convex (and not linear).
yes, possible, can be convex and not linear at any W1, where the elements of W1 are non-zeros, and W2 > 0

W1= [1, -1], W2 = [1, 1]

(d) (BONUS; 0pt) The function is neither convex nor concave.
a linear function is considered both convex and concave because the line segment connecting any two points on a linear function lies directly on the function, satisfying the definitions for both concavity and convexity simultaneously.

a function rhat is neither has to look like an S-shape like sigmoid and tanh, whhcic is impossible for aour function to look like, so answer is impossible


x --> (1, 1)
W1 -> (2, 1)
z1--> (2, 1)
h1 -> (2, 1)
W2 -> (1, 2)
W2h1 -> (1, 1)


f(x;W1,W2) = W2 ReLU(W1x) 

### scalar input and output
W1 = [1, -1]
W2 = [1, -1]
they result in a linear function f(x) = x



##################################################################
##################################################################
##################################################################

6. (BONUS; non-convexity of neural networks; 0pt) Consider a 2-layer ReLU network.
Plot an example to show that the network output is not guaranteed to be convex (or
concave) w.r.t. network parameters. You can pick the network width (although 2
suffices).
In particular, find a fixed input and a linear path in parameter space, and plot the
network output with that fixed input while varying parameters along the linear path.
The plot should be neither convex nor concave.

# check gemini chat for plot


##################################################################
##################################################################
##################################################################

7. (4pt) Logic gate ReLU networks
Hint: ReLU(x) non-linearity is like a ”branching” operation at x = 0. Can you find a
set of weights such that the desired decision boundaries correspond to zero inputs to
ReLU’s?
(a) (OR gate; 2pt) Construct a 2-layer width-2 ReLU network with 2-dimensional
inputs in R
2
such that:
f(x;W1,W2, b1, b2) > 0 ⇐⇒ x1 > 0 OR x2 > 0 (1)
Write out the algebraic formula of f with explicit W1,W2, b1, b2.
Hint: Reminder, the input is not boolean.
(b) (XOR gate; 2pt) Construct a ReLU network with at most 3 layers, each with width
at most 4 and 2-dimensional inputs in R
2
such that:
f(x;W1,W2, b1, b2) > 0 ⇐⇒ (x1 < 0 AND x2 > 0) OR
(x1 > 0 AND x2 < 0) (2)
Write out the algebraic formula of f with explicit weight matrices and bias
vectors.
Hint: This is a more challenging problem. Think about how to implement an
AND gate.
(c) (BONUS; NAND gate and functional completeness; 0pt) Write down a ReLU
network that implements the NAND gate. What does this tell you about the
possibility of representing any boolean function with ReLU networks.


Here is the step-by-step construction of the requested logic gates using ReLU networks. 

### (a) OR Gate

To implement an OR gate for continuous inputs $x \in \mathbb{R}^2$, we need the output to be strictly positive if either $x_1 > 0$ or $x_2 > 0$. Because the ReLU function isolates the positive portion of an input, we can simply sum the ReLU activations of $x_1$ and $x_2$. 

If at least one input is positive, its ReLU will be positive, making the sum positive. If both are negative or zero, their ReLUs are zero, making the sum zero.

**Algebraic Formula:**
$$f(x) = \text{ReLU}(x_1) + \text{ReLU}(x_2)$$

To write this in the standard neural network matrix form $f(x) = W_2\text{ReLU}(W_1x + b_1) + b_2$, we define a 2-layer network with a hidden width of 2:

* **Layer 1 Weights and Biases:**
    $$W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
* **Layer 2 Weights and Biases:**
    $$W_2 = \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad b_2 = 0$$

**Explicit Full Formula:**
$$f(x; W_1, W_2, b_1, b_2) = \begin{bmatrix} 1 & 1 \end{bmatrix} \text{ReLU} \left( \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} \right) + 0$$

---

### (b) XOR Gate

The XOR condition requires $f(x) > 0$ if and only if $x_1$ and $x_2$ have opposite signs (meaning the input lies strictly in the 2nd or 4th quadrant of the Cartesian plane). 

We can achieve this elegantly with a single hidden layer (a 2-layer network) of width 3 by using the following function:
$$f(x) = \text{ReLU}(x_1) + \text{ReLU}(x_2) - \text{ReLU}(x_1 + x_2)$$

Let's verify how this perfectly isolates the XOR regions without needing a 3rd layer:
* **Quadrant 1 ($x_1 > 0, x_2 > 0$):** $f(x) = x_1 + x_2 - (x_1 + x_2) = 0$. (Fails, as desired)
* **Quadrant 3 ($x_1 \le 0, x_2 \le 0$):** $f(x) = 0 + 0 - 0 = 0$. (Fails, as desired)
* **Quadrant 4 ($x_1 > 0, x_2 \le 0$):** * If $x_1 + x_2 > 0$, $f(x) = x_1 + 0 - (x_1 + x_2) = -x_2$. Since $x_2$ is negative, $-x_2 > 0$.
    * If $x_1 + x_2 \le 0$, $f(x) = x_1 + 0 - 0 = x_1$. Since $x_1$ is positive, $x_1 > 0$.
    * *(Quadrant 2 works identically by symmetry).*

**Algebraic Formula:**
$$f(x) = \text{ReLU}(x_1) + \text{ReLU}(x_2) - \text{ReLU}(x_1 + x_2)$$

To express this in explicit matrix form $f(x) = W_2\text{ReLU}(W_1x + b_1) + b_2$:

* **Layer 1 Weights and Biases (Width 3):**
    $$W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
* **Layer 2 Weights and Biases:**
    $$W_2 = \begin{bmatrix} 1 & 1 & -1 \end{bmatrix}, \quad b_2 = 0$$

**Explicit Full Formula:**
$$f(x; W_1, W_2, b_1, b_2) = \begin{bmatrix} 1 & 1 & -1 \end{bmatrix} \text{ReLU} \left( \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} \right) + 0$$

---

### (c) BONUS: NAND Gate and Functional Completeness

When discussing standard boolean functions, we evaluate discrete boolean inputs where $x \in \{0, 1\}^2$. A NAND gate outputs $1$ (or a value $> 0$) for all inputs except when both $x_1 = 1$ and $x_2 = 1$.

We can implement this with a single ReLU neuron using a negative sum and a positive bias:
$$f(x) = \text{ReLU}(-x_1 - x_2 + 1.5)$$

* If $x = (1, 1)$, $f(x) = \text{ReLU}(-0.5) = 0$.
* If $x = (1, 0)$ or $(0, 1)$, $f(x) = \text{ReLU}(0.5) = 0.5 > 0$.
* If $x = (0, 0)$, $f(x) = \text{ReLU}(1.5) = 1.5 > 0$.

**Implications for Functional Completeness:**
In boolean logic, the NAND gate is a **universal gate** (functionally complete). This means that any boolean function, no matter how complex, can be constructed entirely out of interconnected NAND gates. 

Because we have just demonstrated that a ReLU network can perfectly simulate a NAND gate, **it proves that a sufficiently large ReLU neural network can represent any arbitrary boolean function.** This is a foundational step in understanding why neural networks act as universal approximators.


##################################################################
##################################################################
##################################################################

Backpropagation (3pt)
8. (3pt) Let W denote a d×d real matrix, and consider the following system of equations:
y = Wx (3)
u = ReLU(y) (4)
v = u + Wu (5)
L =
1
2
∥v∥
2
2
. (6)
Note that x, y, u and v must all be vectors in R
d
for these equations to make sense.
Since ∥v∥
2
2 denotes the standard squared Euclidean norm of vector v, L is a scalar.
(a) (1pt) Show that ∂L
∂Wij
=
X
d
m=1
vm ·
∂vm
∂Wij
.
In a similar manner to part (a), one may derive the following additional relations:
•
∂vm
∂Wij
=
∂um
∂Wij
+ δim · uj +
X
d
l=1
Wml
∂ul
∂Wij
.
•
∂yk
∂Wij
= δikxj
,
where the “Kronecker delta” is given by δik =
(
1 if i = k;
0 if i ̸= k.
•
∂ul
∂yk
= δlk · Θ(yk),
where Θ denotes the Heaviside step function given by Θ(yk) = (
1 if yk ≥ 0;
0 if yk < 0.
(b) (2pt) Let ∂L
∂W denote the matrix with entries (
∂L
∂W)ij =
∂L
∂Wij
. Using the given
relations and your answer to part (a), show that:
∂L
∂W
= v ⊗ u + diag(Θ(y))(I + W⊤)v ⊗ x,
where ⊗ is the outer product and diag shapes its input into a diagonal matrix.
The advantage of this kind of expression for ∂L
∂W is that it is easy to code up in
PyTorch, and makes efficient use of matrix multiplication primitives, which have
highly optimized, parallelized implementations on GPU.
4
Homework 1
6.S898 Deep Learning
Fall 2024
PyTorch (0pt)
9. (NOT GRADED; 0pt) Complete PyTorch tutorial colab notebooks here. Before
proceeding with the following section, you should at least complete the notebooks
(”Tensor Arithmetic” and ”Network Modules”).
CIFAR-10 Classification (12pt)
In this section, we are going to work on this colab notebook to train a network for classifying
a handwritten digit dataset, CIFAR-10 [Krizhevsky, 2009, Torralba et al., 2008].
The following questions 9-15 are stated in detail in the colab notebook. Please include your
added lines of code, text output, and any plots to them in the same pdf submission.
To download a plot from colab, hold shift while you right click on the image.
Note: A lot of skeleton code is provided to you already. Make sure to read through and
understand them. We will provide less skeleton code in future assignments as you get more
used to deep learning code structures.
10. (Building neural networks; 4pt) Complete the incomplete forward and backward
definitions of the module classes, each using ≤ 5 lines of code. We expect this question
to take more time, as you are being asked to derive and implement the backward pass
for multiple components. Hint: Recall, for linear layers, the forward pass takes the
form: out = Wx + b, and the backward pass requires us to know dL
dout,
dL
dx
,
dL
db
. ReLU
and Loss layers can be similarly computed.
(a) (1pt) Linear.forward
(b) (1pt) Linear.backward
(c) (1pt) ReLU.backward
(d) (1pt) CrossEntropyLoss.backward
11. (Training loop; 2pt) Complete the missing parts in train epoch and evaluate functions, each using ≤ 5 lines of code.
12. (Training curve 1; 2pt) Train a model for 30 epochs and plot the learning curves.
Comment on any interesting observation from the plot.
13. (Universal approximation; 1pt) Neural networks are universal approximators, which
means that we can always find a network that fits the training set. Why do you think
that we didn’t get perfect training accuracy? Write down some ideas for improving
training accuracy. Is a perfect training accuracy all we need?
5
Homework 1
6.S898 Deep Learning
Fall 2024
14. (Data augmentation; 1pt) Write code to create training and validation datasets, where
only the training set has a random cropping augmentation (specifications in colab).
Visualize the effect of the augmentation.
Provide both your code (≤ 5 lines) and your visualization.
15. (Training curve 2; 2pt) Train a model on the new training dataset and plot the learning
curves. Comment on any difference you observe from the previous curves.
References
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report,
University of Toronto, 2009.
Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large
data set for nonparametric object and scene recognition. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2008.
6
MIT OpenCourseWare
https://ocw.mit.edu
6.7960 Deep Learning
Fall 2024
For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms