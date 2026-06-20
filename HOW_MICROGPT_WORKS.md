# How `microgpt.py` Works

This is a guide to the ideas inside [`microgpt.py`](microgpt.py), not a guide to its Python syntax. The program is small enough that every important mechanism of a GPT is visible: the training data, tokenizer, parameters, forward pass, attention, loss, backpropagation, optimizer, checkpoint, and generation loop.

The model is tiny and works with names one character at a time, but its learning problem is fundamentally the same one used to train much larger language models:

> Given everything seen so far, what is likely to come next?

For this model, “everything seen so far” is at most 16 characters and “what comes next” is one of 26 lowercase letters or a special boundary token. For a production language model, the context may contain thousands of word pieces and the vocabulary may contain tens of thousands of tokens. The scale changes dramatically. The core objective does not.

## 1. The entire system in one picture

Training repeatedly performs this loop:

```text
name from input.txt
        ↓
characters converted to token IDs
        ↓
GPT predicts a probability for every possible next token
        ↓
compare the prediction with the actual next token
        ↓
compute a loss: one number measuring how surprised the model was
        ↓
backpropagation determines how every parameter contributed to that loss
        ↓
Adam slightly adjusts all parameters to make similar predictions better next time
```

Generation uses the learned model in a second loop:

```text
start token
        ↓
predict probabilities for the next character
        ↓
sample one character from those probabilities
        ↓
feed that character back into the model
        ↓
repeat until the model emits the boundary token or reaches 16 characters
```

There is no separate rule saying that `q` often follows a vowel, that names tend to end in certain sounds, or that some letter combinations are implausible. The model has to discover useful regularities by becoming less wrong at next-character prediction.

## 2. What the model is really learning

Suppose the training name is `emma`. The model is trained on these five predictions:

| What the model has seen | Correct next token |
|---|---|
| start of name | `e` |
| `e` | `m` |
| `em` | `m` |
| `emm` | `a` |
| `emma` | end of name |

The model does not directly store a table containing those full text prefixes. It has only 4,192 adjustable numbers, called **parameters**, shared across every name and every position. Training pressures those numbers to represent reusable patterns: which letters commonly begin names, which characters follow particular sounds, what long-range combinations occur, and when a name is likely to end.

This distinction matters. Memorizing a list would let the program repeat the training names. Learning shared statistical structure lets it assign sensible probabilities to prefixes it has never seen and generate new name-like strings.

The result is still a probability model, not a database of grammar rules. It may learn something we would describe as “`x` is unusual at the start of an English name,” but internally that knowledge is spread across embeddings, attention weights, and MLP weights. The human sentence is our interpretation of a pattern encoded in many numbers.

## 3. The dataset: many tiny documents

The program reads non-empty lines from `input.txt`. Each line is treated as a separate document, which in this dataset means a name. If the file is absent, the script downloads a names dataset.

This is the code that creates that dataset:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

After this runs, `docs` is simply a list such as `['emma', 'olivia', ...]`. The model never sees the file directly. Everything downstream works with this in-memory list of documents.

The documents are shuffled once. A fixed random seed makes the shuffle, parameter initialization, and later sampling reproducible on the same Python implementation. During training, step `s` selects document `s % len(docs)`.

The default run has 1,000 training steps. Because the names dataset contains far more than 1,000 documents, a default run sees only the first 1,000 names in the shuffled order, one name per step. It is not 1,000 complete passes over the dataset. This is enough to demonstrate learning, but it is much less training than a serious model would receive.

Why use one document per step? Simplicity. Real training normally groups many sequences into a batch so modern hardware can process them in parallel and so each update averages evidence from more examples. Here, one name keeps the complete computation graph small enough to build from individual scalar operations in pure Python.

## 4. Tokenization: turning text into a finite prediction problem

A neural network operates on numbers, not characters. The tokenizer builds a sorted list of all distinct characters in the dataset:

```text
abcdefghijklmnopqrstuvwxyz
```

Each character's position in that list becomes its token ID: `a` is 0, `b` is 1, and so on. A special token named `BOS` gets ID 26, giving the default model a vocabulary of 27 tokens.

The vocabulary is constructed here:

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

`''.join(docs)` temporarily joins all names so `set(...)` can find every distinct character. Sorting makes the mapping deterministic. Because `BOS` is assigned the first ID after the real characters, it cannot collide with any character token.

`BOS` means "beginning of sequence," but the code deliberately uses the same token at both boundaries. The name `emma` becomes conceptually:

```text
BOS, e, m, m, a, BOS
```

The first `BOS` asks the model to predict the first character. The final `BOS` teaches it when a name should stop. Reusing one token for both jobs is sufficient because its meaning is determined by context: before any letters it starts a name; after some letters it ends one.

The conceptual sequence above is built during each training step:

```python
doc = docs[step % len(docs)]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

For `emma`, the list comprehension converts each letter to its integer position in `uchars`. The surrounding list additions put the boundary token on both sides.

This is a **character-level tokenizer**. It has two major consequences:

- The vocabulary is tiny and easy to understand.
- The model must build words from many steps and learn spelling patterns character by character.

Large language models usually use subword tokens. A common word may be one token, while an unusual word is split into several pieces. Subwords make sequences shorter and allow the model to reuse meaningful fragments. Character tokenization is slower per word, but ideal for exposing the algorithm.

The tokenizer is learned only in the weak sense that its vocabulary is derived from the dataset. It does not learn clever character groupings. It also cannot represent a character that was absent from the training data.

## 5. Parameters: where the learned behavior lives

The model's behavior is controlled by matrices of initially random numbers. Every number is wrapped in a `Value` object so the program can later calculate its gradient.

The main dimensions are:

| Setting | Value | Meaning |
|---|---:|---|
| `n_layer` | 1 | One attention-plus-MLP Transformer layer |
| `n_embd` | 16 | Each token is represented by 16 working numbers |
| `block_size` | 16 | The model can process at most 16 positions |
| `n_head` | 4 | Attention is split into four heads |
| `head_dim` | 4 | Each head works with four of the 16 dimensions |
| `vocab_size` | 27 | 26 letters plus the boundary token |

Those dimensions are declared directly:

```python
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
```

`head_dim` is derived rather than independently chosen because the four attention heads must divide the 16-dimensional state evenly: `16 // 4 = 4` dimensions per head.

The learned matrices are:

| Matrix | Shape | Purpose |
|---|---:|---|
| `wte` | 27 x 16 | Token embeddings |
| `wpe` | 16 x 16 | Position embeddings |
| `attn_wq` | 16 x 16 | Produces attention queries |
| `attn_wk` | 16 x 16 | Produces attention keys |
| `attn_wv` | 16 x 16 | Produces attention values |
| `attn_wo` | 16 x 16 | Mixes the attention heads |
| `mlp_fc1` | 64 x 16 | Expands the state for the MLP |
| `mlp_fc2` | 16 x 64 | Compresses the MLP result back to 16 values |
| `lm_head` | 27 x 16 | Produces one score per possible next token |

Together these contain 4,192 trainable scalar parameters.

The matrices begin as small random values:

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
```

The matrix helper takes the desired number of output rows first and input columns second. Every entry is a `Value`, not a plain float, because every learned number must participate in backpropagation.

Random initialization is necessary. If every parameter began with the same value, many neurons would receive identical signals and remain identical; the model would waste most of its capacity. Small random differences break that symmetry. Training then turns arbitrary initial numbers into useful behavior.

A parameter by itself usually has no clean interpretation. Knowledge lives in coordinated patterns across parameters. This is similar to ordinary source code in one limited sense: the meaning of one character in a program depends on its role in the larger program. Unlike source code, the learned "program" is a smooth numerical computation discovered by optimization rather than written as explicit rules.

## 6. Embeddings: giving a token a useful internal identity

A token ID such as 4 for `e` is just an arbitrary label. Treating token IDs as magnitudes would falsely imply that `f` is numerically more than `e`. Instead, `wte` is a lookup table. Each token selects one row containing 16 learned numbers. That row is the token's **embedding**.

You can think of an embedding as a small working description of a token. At initialization it is meaningless. During training it may develop directions that help distinguish vowels from consonants, common starters from common endings, or other useful properties. These properties need not line up neatly with individual dimensions. A concept can be distributed across many dimensions, and one dimension can participate in several concepts.

The same character can mean different things at different positions. The first `a` in a name and an `a` near the end should not be processed identically. `wpe` therefore stores a separate 16-number embedding for each position from 0 through 15.

The model adds the token embedding and position embedding:

```text
current state = meaning of this token + information about where it appears
```

The beginning of `gpt` performs exactly those lookups and that addition:

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

Indexing `wte` selects the row for the current character. Indexing `wpe` selects the row for the current position. `zip` pairs their corresponding 16 components so they can be added component by component. The result, `x`, is the model's first contextual representation of this token.

Addition works because both are represented in the same 16-dimensional workspace. Training jointly shapes these tables so their sum is useful downstream.

## 7. A brief guide to vectors and matrices here

The word **vector** sounds more exotic than the object in this program. It is simply a fixed-length list of numbers. The model's state `x` is a vector of 16 numbers.

A matrix is a table of numbers that transforms one vector into another. Each output number is a weighted sum of the input numbers. The `linear` function performs this operation.

For a tiny example, imagine an input with two numbers:

```text
x = [3, 5]
```

One matrix row `[0.2, -0.4]` produces:

```text
0.2 x 3 + (-0.4) x 5 = -1.4
```

Sixteen rows produce sixteen output numbers. Each row learns a different mixture of the input. A linear transformation can copy, suppress, combine, or contrast information carried by different dimensions.

The implementation is compact:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

For each row `wo` in the weight matrix, the inner expression multiplies corresponding weights and inputs and sums them. The outer list collects one result per row. The shape of `w` therefore determines how many numbers come out.

Calling this operation "linear" is less important than understanding its role: the weight matrix is a learned routing and mixing table.

## 8. RMS normalization: keeping the signal at a workable scale

As a state passes through many weighted sums, its numbers can grow, shrink, or become erratic. `rmsnorm` measures their typical squared size and rescales the whole vector so its overall magnitude is near one.

Conceptually:

```text
typical size = square root of the average squared component
normalized component = original component / typical size
```

The code expresses that calculation as follows:

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

`ms` means mean square. Raising it to `-0.5` means taking one divided by its square root, so multiplying by `scale` is equivalent to dividing by the root-mean-square size.

The small `1e-5` term prevents division by zero and avoids unstable behavior when all values are extremely small.

RMS normalization does not erase the pattern across dimensions. If one component is larger than another, that relationship remains. It mainly controls the overall volume of the signal while preserving its shape.

Why is that useful? The same learned weights must work across many tokens, positions, and training stages. Stable input scale makes their job easier and makes optimization less fragile. The code normalizes before attention and before the MLP, following the common "pre-norm" Transformer arrangement. It also normalizes once immediately after combining the token and position embeddings.

## 9. Attention: retrieving relevant information from the past

Attention is the mechanism that lets the current position use earlier positions. At each layer and position, the model converts the current 16-number state into three new 16-number vectors:

- A **query** asks what information the current position is looking for.
- A **key** describes what each position offers or how it should be matched.
- A **value** contains the information that can be retrieved from that position.

These are not human-readable questions and descriptions. They are learned numerical representations. The query and keys are used to decide **where to look**; the values determine **what is obtained**.

Inside the Transformer layer, three learned linear transformations create them from the same normalized state:

```python
x = rmsnorm(x)
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(v)
```

The current query is used immediately. The current key and value are saved alongside those from earlier positions so this and later queries can attend to them.

### 9.1 The key/value cache

The model is called once for each position, moving from left to right. Every new key and value is appended to a cache. When processing the current character, the cache contains that position and all previous positions, but no future positions.

The caches are created once per document, outside the position loop:

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
```

They are passed back into `gpt` on every position. Because the loop advances left to right, the cache's contents are also chronological. Creating fresh caches for each document prevents one name from attending to another name.

This gives the model **causal attention**: a prediction cannot cheat by looking at the character it is supposed to predict. Production Transformer training often computes all positions in parallel and applies a triangular mask to hide the future. This implementation gets the same causal restriction naturally from its sequential loop.

The cache also avoids recomputing keys and values for previous positions during generation. Production inference systems rely heavily on this same idea, usually called a KV cache.

### 9.2 Matching queries with keys

For each earlier position, the model multiplies corresponding query and key components and adds the products. This dot product is a compatibility score:

```text
high score  → this past position looks relevant to the current query
low score   → it looks less relevant
```

The score is divided by the square root of the head dimension. Without this scaling, dot products tend to grow as vectors gain more components. Large scores would make softmax excessively decisive and gradients less useful. Scaling keeps the range manageable.

Softmax turns all compatibility scores into positive weights that sum to one. The attention output is a weighted average of the value vectors. A position with weight 0.7 contributes much more than one with weight 0.02.

Here is one head's complete retrieval calculation:

```python
q_h = q[hs:hs+head_dim]
k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
v_h = [vi[hs:hs+head_dim] for vi in values[li]]

attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
attn_weights = softmax(attn_logits)
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

The first three lines isolate this head's four dimensions. `attn_logits` creates one query-key match score per available position. `attn_weights` converts those scores into a probability-like distribution. Finally, each component of `head_out` is the weighted mixture of that component from every cached value.

So attention performs a soft, content-dependent lookup:

1. Describe the current need with a query.
2. Compare that query with every available key.
3. Convert the matches into attention weights.
4. Blend the corresponding values according to those weights.

It says “soft” because the model normally mixes information from several positions instead of selecting exactly one.

### 9.3 Why multiple heads?

The 16 dimensions are divided into four heads of four dimensions each. Each head computes its own attention weights and value mixture.

Separate heads allow different retrieval patterns at the same position. One head might benefit from focusing on the immediately previous character; another might use the first character; another might track a broader spelling pattern. These descriptions are possibilities, not hard-coded jobs. Training determines whether the heads specialize and what they specialize in.

The four head outputs are concatenated back into 16 numbers. `attn_wo` then mixes them, allowing information discovered by different heads to interact.

The enclosing loop performs this split-and-recombine operation:

```python
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    # The per-head attention calculation produces head_out here.
    x_attn.extend(head_out)
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

`extend` appends all four numbers rather than appending the four-number list as one nested item. After four heads, `x_attn` is therefore a flat 16-number vector ready for the output projection. The comment marks the calculation shown in the preceding snippet.

An important detail is that this implementation first creates full 16-dimensional queries, keys, and values and then slices each into four heads. That is mathematically equivalent to giving each head its own relevant sections of the projection matrices.

## 10. Residual connections: preserving an information highway

After attention, its output is added to the state that existed before attention:

```text
new state = old state + attention result
```

In the source, preserving the old state and adding it back looks like this:

```python
x_residual = x
x = rmsnorm(x)
# Attention transforms x here.
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]
```

The attention path is allowed to change `x`, but `x_residual` still refers to the state from before that path. The final component-wise addition creates the information highway.

The MLP uses the same pattern:

```text
new state = old state + MLP result
```

These are **residual connections**. They let each block contribute a correction rather than reconstructing the entire state. If attention has nothing useful to add for a particular input, it can learn an output near zero and leave the old state mostly unchanged.

Residual paths also make training easier. During backpropagation, they provide a direct route for gradient information to reach earlier computations. Deep neural networks without such routes are much harder to optimize because every signal must survive a long chain of transformations.

Although this model has only one layer, it retains the architecture that makes much deeper Transformers practical.

## 11. The MLP: thinking independently at each position

Attention moves information between positions. The MLP then processes the resulting information at the current position.

It performs three steps:

1. Expand the 16-number state to 64 numbers.
2. Apply ReLU, replacing negative values with zero.
3. Project the 64 numbers back down to 16.

Those steps, including the residual connection around them, are visible together:

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

`mlp_fc1` has 64 rows, so the first `linear` call returns 64 numbers. `mlp_fc2` has 16 rows, so the second returns the state to 16 numbers before it is added to the residual.

Why expand? A larger intermediate workspace gives the model room to detect and combine many features. The first matrix can create candidate features such as particular combinations of token identity, position, and retrieved context. ReLU makes the computation nonlinear: candidates with negative activation are shut off, while positive ones pass through. The second matrix combines the active features into a useful update to the state.

Without a nonlinear operation, stacking linear transformations would collapse into one linear transformation. The model could mix information but could not build the same kind of conditional behavior. ReLU lets the network behave differently depending on which learned features are active.

The MLP is applied separately at every position, using the same learned weights each time. It does not retrieve earlier states directly; that is attention’s job. A useful rough division of labor is:

- Attention communicates across positions.
- The MLP computes on the information available at one position.

## 12. From the final state to a next-token prediction

After the Transformer layer, `lm_head` converts the final 16-number state into 27 numbers, one for every vocabulary token. These numbers are **logits**.

The last line of `gpt` performs that projection:

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

A logit is an unconstrained score. It is not yet a probability: it may be negative, positive, large, or small. Only relative differences matter. Softmax converts logits into probabilities by exponentiating them and dividing by their total.

If three logits produced probabilities `[0.70, 0.20, 0.10]`, the model would be expressing a 70% preference for the first token. All 27 probabilities sum to one.

Before exponentiating, the code subtracts the largest logit. This changes none of the final probabilities because it shifts every logit equally, but it prevents `exp` from overflowing on large numbers. This is a numerical-stability technique, not a modeling choice.

That stable softmax is implemented here:

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Every exponentiated score is positive, and dividing by `total` makes the outputs sum to one. `max_val` is read as an ordinary number because it is used only as a shared numerical offset; the model does not need to learn through the choice of which logit was largest.

The output matrix is not tied to the token embedding matrix. Some GPTs reuse one matrix for both token input and output scoring, reducing parameter count and encouraging a shared representation. This small implementation keeps them separate, which makes the two roles explicit.

## 13. Loss: turning “wrong” into one useful number

Training needs a precise answer to “how bad was this prediction?” For the correct next token, the code takes:

```text
loss = -log(probability assigned to the correct token)
```

This is cross-entropy loss for a single target, also called negative log-likelihood.

Some examples:

| Probability assigned to the correct token | Loss |
|---:|---:|
| 0.90 | about 0.11 |
| 0.50 | about 0.69 |
| 0.10 | about 2.30 |
| 0.01 | about 4.61 |

The loss heavily penalizes confident mistakes. Assigning 1% probability to the truth is much worse than assigning 50%. This is exactly what we want from a model that claims to produce probabilities: it should not be confidently wrong.

For one name, the program computes a loss at every position and averages them. Every character prediction therefore contributes equally to the training step, including prediction of the ending boundary.

This happens directly in the position loop:

```python
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = (1 / n) * sum(losses)
```

`target_id` is always one place ahead of `token_id`. Indexing `probs[target_id]` selects exactly the probability assigned to the truth. The negative logarithm turns that probability into the position's loss, and the final line averages across positions.

The block size is 16. The code trains on at most 16 next-token predictions from a document. In the supplied dataset, the longest name has 15 characters, so 16 predictions cover its letters plus its ending token. With longer input documents, anything beyond this window would be ignored by the current training loop.

Loss is a training instrument. A lower average loss means the model assigned more probability to the observed next tokens. It does not by itself guarantee that generated names will be diverse, pleasant, or novel. Those qualities depend on both the learned distribution and how it is sampled.

## 14. Autograd: assigning numerical blame

The most unusual part of this script is that it implements automatic differentiation from scratch. Libraries such as PyTorch normally do this invisibly and efficiently. Here, every arithmetic result is a `Value` node containing:

- `data`: the numerical result of the forward calculation;
- `grad`: how much the final loss changes if this value changes slightly;
- `_children`: the earlier values used to create it;
- `_local_grads`: how the result locally changes with each child.

The constructor stores that information, and each operation records its own local derivatives:

```python
def __init__(self, data, children=(), local_grads=()):
    self.data = data
    self.grad = 0
    self._children = children
    self._local_grads = local_grads

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(
        self.data * other.data,
        (self, other),
        (other.data, self.data),
    )
```

For addition, changing either input by a small amount changes the output by the same amount, so both local gradients are 1. For multiplication, the sensitivity to one input is the value of the other input.

Imagine a simple computation:

```text
c = a × b
loss = c + d
```

If `a` increases slightly, `c` changes by roughly `b` times that amount. The local derivative of `c` with respect to `a` is therefore `b`. Because `loss` includes `c` directly, the influence continues to the loss.

Backpropagation applies this logic through the entire graph. It starts at the loss with gradient 1: a one-unit increase in the loss is, trivially, a one-unit increase in itself. It then walks backward. At each operation:

```text
influence on child = influence on current result × local influence of child
```

This multiplication of linked influences is the chain rule. No advanced calculus is needed to understand its practical meaning: if A affects B, and B affects the loss, multiply those two sensitivities to find how A affects the loss.

The backward traversal implements exactly that rule:

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

`build_topo` orders the graph so a result is processed before the values that created it during the reverse pass. The final line is the chain rule in code: upstream influence `v.grad` multiplied by the operation's `local_grad`.

Gradients are added rather than replaced because one value may affect the loss through several routes. A parameter is reused at every position, and each use supplies evidence about how it should change. Accumulation combines that evidence.

The graph is sorted so every node is processed only after all nodes depending on it have passed their gradients backward. When `loss.backward()` finishes, every parameter’s `grad` estimates the slope of the loss with respect to that parameter.

A gradient does not directly say the ideal parameter value. It says which direction locally increases the loss and how sensitive the loss is. Moving a small distance in the opposite direction should reduce the loss for the current example.

## 15. Why the forward pass builds the backward pass

During the forward pass, operations use `Value` objects rather than plain floating-point numbers. Adding, multiplying, taking a logarithm, or applying ReLU produces both a numeric answer and a record of how that answer was made.

By the time the program computes the loss, it has built a large graph connecting that loss all the way back to every parameter involved in the name’s predictions. The graph is dynamic: it is the record of the actual operations performed for this particular name.

This explains a detail that can otherwise look odd: early token states stored in the attention cache remain part of the computation graph. A later prediction can attend to an earlier value, so its loss can send gradients backward through that attention connection and into the computations that produced the earlier key and value.

After the optimizer update, the graph is no longer needed. The next training step builds a fresh graph for the next name.

## 16. Adam: turning gradients into parameter updates

The simplest optimizer would subtract a fixed multiple of each gradient from its parameter. Adam improves on this by keeping two running summaries for every parameter:

- `m`, a moving average of recent gradients, estimates the persistent direction.
- `v`, a moving average of squared gradients, estimates the recent scale of updates.

The parameter update is roughly:

```text
parameter -= learning_rate × direction / typical_gradient_size
```

This gives each parameter an adaptive step size. Parameters receiving consistently large gradients are normalized differently from parameters receiving small gradients. The first-moment estimate also smooths noisy example-to-example directions.

Both moving averages begin at zero, which biases their early values toward zero. `m_hat` and `v_hat` correct that startup bias. The tiny `eps_adam` term prevents division by zero.

The complete parameter update is:

```python
lr_t = learning_rate * (1 - step / num_steps)
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

The only line that changes the learned model is the subtraction from `p.data`. All surrounding calculations decide the size and direction of that change.

The learning rate begins at 0.01 and decreases linearly toward zero across 1,000 steps. Early updates can make broad progress; later updates become more cautious. Because the final step still uses a small positive rate of `0.01 / 1000`, it approaches zero without exactly reaching it during the loop.

After updating a parameter, the code resets its gradient to zero. This is essential. Gradient accumulation was useful within one computation graph, but carrying it into the next training step would accidentally combine unrelated steps.

The chosen Adam coefficients and learning rate are practical values for this demonstration, not universal constants. Larger models tune optimization settings carefully because training stability and final quality depend heavily on them.

## 17. What one complete training step does

Putting the pieces together, one step is:

1. Select one shuffled name.
2. Convert its characters to IDs and place the boundary token at both ends.
3. Create empty key/value caches.
4. For each position, run `gpt` on the current token using only cached past and current positions.
5. Turn the output logits into next-token probabilities.
6. Measure negative log-probability of the actual next token.
7. Average all position losses into one scalar.
8. Backpropagate from that scalar to obtain gradients for all participating parameters.
9. Use Adam to update every parameter.
10. Clear gradients and continue with the next name.

The key idea is that training never tells the model what linguistic feature to detect. It supplies only input prefixes and correct next tokens. Backpropagation and optimization search for internal features that make those predictions easier.

## 18. Checkpointing: freezing the learned numbers

After training, `save_run` creates a timestamped directory under `saved_runs/` and writes `model.json`. The checkpoint contains:

- the architectural settings needed to reconstruct the model;
- the character vocabulary and boundary-token ID;
- every trained matrix as ordinary numeric data;
- a run name and UTC creation time.

The important serialization step converts each trainable `Value` back to its plain numeric data:

```python
payload = {
    "config": {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "BOS": BOS,
    },
    "uchars": uchars,
    "state_dict": {
        name: [[value.data for value in row] for row in mat]
        for name, mat in state_dict.items()
    },
}
```

Inference does not need computation-graph metadata or gradients, so only `value.data` is saved. The config records the shapes needed to interpret those nested lists later.

The checkpoint is the learned model. The Python functions define how to use the numbers, while the JSON file supplies the numbers learned by one run.

The file does **not** contain Adam’s `m` and `v` buffers, the current training step, the shuffled document order, or random-number-generator state. It is sufficient for inference, but not for resuming training in a way exactly equivalent to an uninterrupted run.

JSON is convenient and inspectable, but inefficient for large numerical models. Production checkpoints use compact binary formats and often split parameters across files or machines.

## 19. Inference: making the model generate

Training always provides the correct previous token. Inference has no answer sheet, so the model must consume its own output.

Each sample begins with:

```text
current token = BOS
generated text = empty
KV cache = empty
```

The model predicts logits for the next token. After temperature adjustment and softmax, the program randomly samples one token according to the probabilities. If it samples `BOS`, the name is complete. Otherwise the character is appended, becomes the next input token, and the process repeats.

The generation loop is:

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
token_id = BOS
sample = []
for pos_id in range(block_size):
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax([l / temperature for l in logits])
    token_id = random.choices(
        range(vocab_size),
        weights=[p.data for p in probs],
    )[0]
    if token_id == BOS:
        break
    sample.append(uchars[token_id])
```

Notice that `token_id` serves as both output and the next loop iteration's input. That single reassignment is the feedback connection that makes generation autoregressive.

This is **autoregressive generation**: outputs are produced one at a time, and every output becomes part of the input for later outputs. A poor early choice can change the entire continuation. There is no hidden plan for the completed name.

Generation stops after at most 16 positions even if the model never emits `BOS`. The context limit is therefore also a hard maximum generated length in this script.

### Temperature

Before softmax, every logit is divided by `temperature`, which defaults to 0.5.

In code, temperature appears immediately before softmax:

```python
probs = softmax([l / temperature for l in logits])
```

Dividing every logit by a value below 1 increases the distances between logits before softmax sees them. That is why a temperature of 0.5 makes the resulting probabilities sharper.

- A temperature below 1 exaggerates logit differences. Likely tokens become more likely and unlikely tokens become less likely. Output is more conservative.
- A temperature of 1 samples from the model’s learned distribution without adjustment.
- A temperature above 1, although the comment presents the intended range as up to 1, would flatten the distribution and produce more varied, risky choices.
- As temperature approaches zero, sampling approaches always choosing the highest-logit token.

Temperature does not make the model know more or less. It changes how boldly we sample from what it already knows.

Sampling rather than always choosing the most likely token is what allows multiple different names from the same model. Greedy selection can become repetitive and ignores plausible alternatives. Random sampling introduces diversity while still respecting learned probabilities.

## 20. Why this is genuinely a GPT

The acronym expands to **Generative Pre-trained Transformer**:

- **Generative:** it models and samples sequences one token at a time.
- **Pre-trained:** its parameters are learned from a corpus through next-token prediction before any later use. Here, there is no separate fine-tuning stage, but the training phase plays the pretraining role.
- **Transformer:** its main learned block combines causal self-attention, an MLP, normalization, and residual connections.

The program is not merely “inspired by” a GPT. It implements the essential GPT algorithm in a deliberately tiny form. Its output quality and capability are limited by data, scale, tokenization, and training time, not by the absence of the central mechanism.

## 21. What has been simplified

The script’s opening claim that everything else is efficiency is directionally useful, but production models also contain modeling and systems refinements that affect stability and quality. Important simplifications include:

- **Character tokens:** production systems usually use learned subword vocabularies.
- **One sequence at a time:** real training uses large batches and parallel hardware.
- **Scalar arithmetic:** every number is a Python object; tensor libraries process huge arrays in optimized kernels.
- **One layer and width 16:** useful language models are vastly deeper and wider.
- **Context length 16:** modern contexts are measured in thousands or more tokens.
- **One training pass over 1,000 names:** large models consume enormous corpora over many optimization steps.
- **No biases:** many linear layers elsewhere include learned offsets.
- **RMSNorm without learned scale:** common implementations include a learned per-dimension multiplier.
- **ReLU:** modern language models often use GELU, SwiGLU, or related activations.
- **Learned absolute positions:** many current models use rotary positional information or other approaches.
- **No regularization or validation split:** there is no dropout and no held-out evaluation of generalization.
- **No optimizer checkpoint:** saved runs support inference, not exact training resumption.

None of these omissions prevents the code from demonstrating the central learning loop. They remove distractions and computational expense.

## 22. Common misconceptions to avoid

### “The model searches the input file when generating.”

It does not. After training, generation needs only the learned matrices and vocabulary. The dataset influenced the parameter values, but there is no name lookup during inference.

### “Attention is a database of previous training examples.”

Here, attention operates over positions in the **current name prefix**, not over the training dataset. Its keys and values are rebuilt for each new sequence. The training corpus is reflected indirectly in the learned projection matrices.

### “The largest probability is the model’s answer.”

The model produces a full distribution over possible next tokens. Sampling chooses one realization from that distribution. Another run can take a different plausible branch.

### “A low loss means the model understood names like a person.”

Low loss means it predicts held examples well under the chosen objective. The internal patterns may be useful and structured, but the objective does not test human understanding, factual grounding, or intent.

### “A gradient explains what a parameter means.”

A gradient only reports local sensitivity for the current loss. Interpretation requires examining behavior across many inputs, and even then knowledge is usually distributed.

### “The generated text is copied unless every name is unique.”

The model can both memorize examples and recombine learned patterns. Novel output is evidence of generalization, but novelty alone does not tell us how much was understood or memorized.

## 23. A useful mental model

Think of the model as a small machine with a 16-number scratchpad.

For each character:

1. The embedding tables place token identity and position onto the scratchpad.
2. Normalization keeps the scratchpad’s scale controlled.
3. Attention reads selected information from earlier scratchpads.
4. A residual connection preserves the existing information while adding the retrieved result.
5. The MLP detects and transforms useful combinations on the current scratchpad.
6. Another residual connection preserves and extends the state.
7. The language-model head scores every possible next character.

During training, the correct next character tells the system how surprising its prediction was. Backpropagation traces that surprise through every operation and assigns a small amount of responsibility to every parameter. Adam uses those responsibilities to adjust the machine. Repeating this process turns random weights into a model of name-like character sequences.

That is the core of GPT training: predict, measure surprise, assign blame, adjust, repeat.

## 24. Reading the source in a productive order

The file is short, but reading it from top to bottom mixes several conceptual layers. A more useful order is:

1. **Dataset and tokenizer:** establish the actual prediction problem.
2. **Training loop:** see the repeated forward-loss-backward-update lifecycle.
3. **`gpt`:** trace how one current token becomes next-token logits.
4. **`linear`, `softmax`, and `rmsnorm`:** understand the primitive model operations.
5. **`Value.backward`:** understand how the loss reaches the parameters.
6. **Parameter initialization:** map each matrix to the role you have now seen.
7. **Inference:** see the same forward model used without targets or gradients.
8. **Checkpointing:** see which parts of a trained run must survive after the process exits.

On a first pass, do not try to interpret individual learned numbers. Follow shapes and responsibilities: 16 values represent the current state, four heads retrieve from the available past, and 27 output scores represent the possible future. Once that flow is clear, the implementation becomes a compact statement of the full algorithm rather than a wall of arithmetic.
