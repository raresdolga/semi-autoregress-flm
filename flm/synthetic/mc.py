"""
Markov chain model to generate synthetic data.

The clean sequence is a first-order Markov chain over K tokens with a sparse
transition matrix (every token has exactly b successors, each with probability
1/b). Noising the one-hot tokens with x_t = lambda_t x_1 + (1 - lambda_t) x_0,
x_0 ~ N(0, I), turns the chain into an HMM, so the exact posterior over clean
tokens is computed with the forward-backward recursions. Everything is done in
log space because the Gaussian emissions blow up as t -> 1.
"""

import math

import torch


class MC:
  """Markov chain with uniform marginals and its exact noisy posterior.

  Token j may be followed by (j + s) mod K for every offset s in S, each with
  probability 1/b, so T(j, k) = 1/b * 1(k in {j + s mod K : s in S}). The
  initial distribution is uniform, which makes every position uniform.

  Args:
    K: Vocabulary size (number of hidden states).
    D: Sequence length (number of positions).
    b: Branching factor, the number of allowed successors of each token,
      1 <= b <= K. b = 1 makes the chain deterministic given its first token,
      b = K makes the tokens independent.
    offsets: Optional set S of b distinct offsets in [0, K). If None, b
      distinct offsets are drawn at random with `seed`.
    seed: Seed used to draw the offsets when `offsets` is None.
  """

  def __init__(self, K, D, b, offsets=None, seed=0):
    if K < 1 or D < 1:
      raise ValueError(f'K and D must be positive, got K={K}, D={D}.')
    if not 1 <= b <= K:
      raise ValueError(f'b must be in [1, K={K}], got {b}.')
    self.K = K
    self.D = D
    self.b = b

    if offsets is None:
      gen = torch.Generator().manual_seed(seed)
      offsets = torch.randperm(K, generator=gen)[:b].sort().values
    offsets = torch.as_tensor(offsets, dtype=torch.long).flatten()
    if offsets.numel() != b or offsets.unique().numel() != b:
      raise ValueError(f'offsets must hold {b} distinct values, got '
                       f'{offsets.tolist()}.')
    if (offsets < 0).any() or (offsets >= K).any():
      raise ValueError(f'offsets must lie in [0, {K}), got {offsets.tolist()}.')
    self.offsets = offsets.tolist()

    self.log_pi = -math.log(K)  # uniform initial distribution
    self.log_T = -math.log(b)   # value of every non-zero entry of T

  @property
  def lambda_t(self):
    """Schedule lambda(t) = t, returned as a function: `mc.lambda_t(t)`."""
    return lambda t: t

  def transition_matrix(self, dtype=torch.float32, device=None):
    """Dense transition matrix T of shape (K, K), T[j, k] = P(k | j)."""
    T = torch.zeros(self.K, self.K, dtype=dtype, device=device)
    j = torch.arange(self.K, device=device)
    for s in self.offsets:
      T[j, (j + s) % self.K] = 1.0 / self.b
    return T

  def sample(self, n, generator=None):
    """Draws n clean sequences of token indices, shape (n, D)."""
    offsets = torch.tensor(self.offsets)
    k = torch.empty(n, self.D, dtype=torch.long)
    k[:, 0] = torch.randint(self.K, (n,), generator=generator)
    for d in range(1, self.D):
      choice = torch.randint(self.b, (n,), generator=generator)
      k[:, d] = (k[:, d - 1] + offsets[choice]) % self.K
    return k

  def sample_xt(self, t, k=None, generator=None):
    """Draws noisy inputs x_t = lambda_t x_1 + (1 - lambda_t) x_0, eq. 1.

    Args:
      t: Times in [0, 1], tensor of shape (B,).
      k: Optional clean token indices of shape (B, D). If None, B clean
        sequences are drawn from the chain.
      generator: Optional torch.Generator for the clean tokens and the noise.

    Returns:
      x_t: Noisy inputs of shape (B, D, K), float tensor with the dtype and
        device of t.
    """
    t = torch.as_tensor(t)
    if not t.is_floating_point():
      t = t.float()
    B = t.shape[0]
    if k is None:
      k = self.sample(B, generator=generator)
    x1 = torch.nn.functional.one_hot(k.to(t.device), self.K).to(t.dtype)
    x0 = torch.randn(x1.shape, generator=generator, dtype=t.dtype)
    lam = self.lambda_t(t).reshape(B, 1, 1)
    return lam * x1 + (1 - lam) * x0.to(t.device)

  def calc_probs(self, x_t, t):
    """Emissions and forward-backward messages for a noisy input.

    Args:
      x_t: Noisy inputs of shape (B, D, K).
      t: Time in [0, 1); a float or a tensor of shape (B,).

    Returns:
      log_phi: log phi_t^d(k), shape (B, D, K).
      log_alpha: log alpha^d(k), shape (B, D, K).
      log_beta: log beta^d(k), shape (B, D, K).
    """
    if x_t.dim() != 3 or x_t.shape[1:] != (self.D, self.K):
      raise ValueError(f'x_t must have shape (B, {self.D}, {self.K}), '
                       f'got {tuple(x_t.shape)}.')
    log_phi = self._log_phi(x_t, t)
    log_alpha = self._forward(log_phi)
    log_beta = self._backward(log_phi)
    return log_phi, log_alpha, log_beta

  def posterior(self, x_t, t):
    """Exact density p_t(x_t) and posterior marginals p_t(k^d = k | x_t).

    Args:
      x_t: Noisy inputs of shape (B, D, K).
      t: Time in [0, 1); a float or a tensor of shape (B,).

    Returns:
      log_px: log p_t(x_t), shape (B,).
      probs: p_t(k^d = k | x_t), shape (B, D, K). This is also the posterior
        mean mu_t^d(x_t), since x_1^d is one-hot.
    """
    _, log_alpha, log_beta = self.calc_probs(x_t, t)
    log_px = torch.logsumexp(log_alpha[:, -1], dim=-1)
    probs = torch.softmax(log_alpha + log_beta, dim=-1)
    return log_px, probs

  def _log_phi(self, x_t, t):
    """log phi_t^d(k) = log N(x_t^d | lambda_t e_k, (1 - lambda_t)^2 I).

    Uses ||x - lam e_k||^2 = ||x||^2 - 2 lam x_k + lam^2. Returns (B, D, K).
    """
    lam = torch.as_tensor(self.lambda_t(t), dtype=x_t.dtype, device=x_t.device)
    if (lam < 0).any() or (lam >= 1).any():
      raise ValueError('lambda_t must lie in [0, 1).')
    lam = lam.reshape(-1, 1, 1)
    var = (1 - lam) ** 2
    sq = (x_t ** 2).sum(-1, keepdim=True) - 2 * lam * x_t + lam ** 2
    return -0.5 * self.K * torch.log(2 * math.pi * var) - sq / (2 * var)

  def _forward(self, log_phi):
    """log alpha^1 = log pi + log phi^1,
    log alpha^{d+1}(k) = log phi^{d+1}(k) + log(1/b)
                         + logsumexp_s [log alpha^d(k - s)].
    """
    alphas = [self.log_pi + log_phi[:, 0]]
    for d in range(1, self.D):
      # roll(a, s)[k] = a[k - s]: the predecessors of k are k - s.
      prev = torch.stack(
          [torch.roll(alphas[-1], s, dims=-1) for s in self.offsets])
      alphas.append(log_phi[:, d] + self.log_T + torch.logsumexp(prev, dim=0))
    return torch.stack(alphas, dim=1)

  def _backward(self, log_phi):
    """log beta^D = 0,
    log beta^d(j) = log(1/b)
                    + logsumexp_s [log phi^{d+1}(j + s) + log beta^{d+1}(j + s)].
    """
    betas = [torch.zeros_like(log_phi[:, -1])]
    for d in range(self.D - 2, -1, -1):
      g = log_phi[:, d + 1] + betas[-1]
      # roll(g, -s)[j] = g[j + s]: the successors of j are j + s.
      nxt = torch.stack([torch.roll(g, -s, dims=-1) for s in self.offsets])
      betas.append(self.log_T + torch.logsumexp(nxt, dim=0))
    return torch.stack(betas[::-1], dim=1)
