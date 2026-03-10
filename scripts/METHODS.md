# Online Pose Estimation Methods: Technical Reference

This document provides a formal treatment of the three online camera pose estimation methods implemented in GOGGLES. All methods produce world-to-camera extrinsics $\mathbf{T}^{w2c}_i \in \mathrm{SE}(3)$ for a temporally-ordered image sequence $\{\mathcal{I}_1, \dots, \mathcal{I}_N\}$.

---

## 1. StreamVGGT with Sliding-Window KV Cache

### 1.1 Architecture

StreamVGGT is an autoregressive vision transformer that processes frames sequentially via cached key–value (KV) attention. The architecture comprises:

- **Backbone**: DINOv2 encoder mapping each image $\mathcal{I}_i$ to patch tokens.
- **Aggregator**: $L = 24$ alternating attention layers. Each layer applies:
  - *Frame attention* (dims $0{:}1023$): self-attention within a single frame's tokens.
  - *Global attention* (dims $1024{:}2047$): causal cross-frame attention over all cached frames.
- **Camera head**: $L_c = 4$ AdaLN-modulated transformer blocks operating on the camera token (index 0) from the aggregator output. Predicts a 9-DOF pose encoding per frame:

$$\hat{\mathbf{p}}_i = (\mathbf{t}_i \in \mathbb{R}^3,\; \mathbf{q}_i \in \mathbb{R}^4,\; f_i \in \mathbb{R}^1)$$

where $\mathbf{t}_i$ is translation, $\mathbf{q}_i$ is a unit quaternion rotation, and $f_i$ is focal length.

### 1.2 KV Cache and Causal Attention

At frame $i$, the aggregator's global attention layers attend over all previously cached frames. Denoting the key and value tensors at layer $\ell$ as $\mathbf{K}^{(\ell)}, \mathbf{V}^{(\ell)} \in \mathbb{R}^{B \times H \times F \times P \times d}$ (batch, heads, frames, patches-per-frame, head dimension), the cache grows along the frame axis $F$ with each new observation.

The camera head maintains a separate KV cache $\mathbf{K}^{(c)}_\ell, \mathbf{V}^{(c)}_\ell \in \mathbb{R}^{B \times H \times F \times 1 \times d}$ with $P = 1$ (camera token only).

### 1.3 Sliding Window

For sequences exceeding available memory, the aggregator KV cache is truncated to a sliding window of $W$ frames. After processing frame $i$:

$$\mathbf{K}^{(\ell)} \leftarrow \mathbf{K}^{(\ell)}[:, :, -W:, :, :], \quad \forall\, \ell \in \{1, \dots, L\}$$

and identically for $\mathbf{V}^{(\ell)}$. This bounds memory at $\mathcal{O}(W \cdot P)$ per layer regardless of sequence length.

**The camera head cache is not truncated.** Its per-frame memory cost is negligible ($P = 1$ token vs. $P \approx 782$ for the aggregator — a factor of $\sim\!4700\times$), and the camera head requires the full history to maintain a consistent global coordinate frame across all predicted poses.

### 1.4 Inference Regimes

The sequence naturally partitions into two regimes:

- **Frames $1$ to $W$** (accumulative): The cache only grows. Each new frame adds context; no information is discarded.
- **Frames $W{+}1$ to $N$** (sliding): Each new frame evicts the oldest. The aggregator's global attention reflects only the most recent $W$ frames, while the camera head retains full history.

---

## 2. DA3 Chunked with SIM(3) Submap Alignment

### 2.1 Overview

This method partitions the sequence into overlapping chunks, runs DA3 batch inference independently per chunk (yielding per-chunk poses, depths, and intrinsics in a local coordinate frame), and aligns adjacent chunks into a common global frame via $\mathrm{SIM}(3)$ estimation on overlapping 3D point maps. The approach is a simplified variant of VGGT-Long (Deng) / DA3-Streaming (ByteDance), omitting robust IRLS weighting and loop closure.

### 2.2 Chunking

Given chunk size $C$ and overlap $O < C$, the sequence is partitioned into chunks with stride $C - O$:

$$\mathcal{C}_k = \{i : s_k \leq i < e_k\}, \quad s_0 = 0, \quad s_{k+1} = s_k + (C - O), \quad e_k = \min(s_k + C,\; N)$$

Adjacent chunks $\mathcal{C}_k$ and $\mathcal{C}_{k+1}$ share $O$ frames in their overlap region $\mathcal{O}_{k,k+1} = \mathcal{C}_k \cap \mathcal{C}_{k+1}$.

### 2.3 Per-Chunk Inference

DA3 batch inference on chunk $\mathcal{C}_k$ yields, in a chunk-local coordinate frame:

- Extrinsics: $\{\mathbf{T}^{(k)}_i \in \mathrm{SE}(3)\}$ for $i \in \mathcal{C}_k$
- Depth maps: $\{D^{(k)}_i \in \mathbb{R}^{H \times W}_+\}$
- Intrinsics: $\{\mathbf{K}^{(k)}_i \in \mathbb{R}^{3 \times 3}\}$
- Confidence maps: $\{c^{(k)}_i \in \mathbb{R}^{H \times W}_+\}$

### 2.4 Depth-to-Point-Cloud Unprojection

For a frame $i$ in chunk $k$, the world-space 3D point at pixel $(u, v)$ is obtained by:

$$\mathbf{x}^{(k)}_{i}(u, v) = \left(\mathbf{T}^{(k)}_i\right)^{-1} \begin{pmatrix} D^{(k)}_i(u,v) \cdot \left(\mathbf{K}^{(k)}_i\right)^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \\ 1 \end{pmatrix}$$

yielding a dense point map $\mathbf{X}^{(k)}_i \in \mathbb{R}^{H \times W \times 3}$ in chunk $k$'s local frame.

### 2.5 SIM(3) Alignment via Umeyama

For overlapping frames $j \in \mathcal{O}_{k,k+1}$, the same image produces point maps $\mathbf{X}^{(k)}_j$ and $\mathbf{X}^{(k+1)}_j$ in the two chunks' respective local frames. Since these are unprojections of the same pixels, points at identical pixel locations $(u, v)$ correspond to the same physical 3D point, giving dense correspondences without feature matching.

High-confidence pixels are selected by thresholding:

$$\mathcal{M}_j = \left\{(u, v) : c^{(k)}_j(u,v) > \tau \;\wedge\; c^{(k+1)}_j(u,v) > \tau\right\}, \quad \tau = 0.1 \cdot \min\!\left(\mathrm{median}(c^{(k)}_j),\; \mathrm{median}(c^{(k+1)}_j)\right)$$

Collecting all confident correspondences across the $O$ overlap frames:

$$\mathcal{S} = \left\{\mathbf{X}^{(k+1)}_j(u,v)\right\}_{j, (u,v) \in \mathcal{M}_j}, \qquad \mathcal{T} = \left\{\mathbf{X}^{(k)}_j(u,v)\right\}_{j, (u,v) \in \mathcal{M}_j}$$

We seek the $\mathrm{SIM}(3)$ transform $(s, \mathbf{R}, \mathbf{t})$ mapping source $\mathcal{S}$ (chunk $k{+}1$'s frame) to target $\mathcal{T}$ (chunk $k$'s frame):

$$\mathbf{p}^{(k)} \approx s\,\mathbf{R}\,\mathbf{p}^{(k+1)} + \mathbf{t}$$

This is solved in closed form by the **Umeyama algorithm**. Given $M$ correspondences $\{(\mathbf{s}_m, \mathbf{t}_m)\}_{m=1}^{M}$:

1. **Centroids**:

$$\bar{\mathbf{s}} = \frac{1}{M}\sum_{m}\mathbf{s}_m, \qquad \bar{\mathbf{t}} = \frac{1}{M}\sum_{m}\mathbf{t}_m$$

2. **Scale** (ratio of RMS spreads):

$$s = \frac{\sqrt{\frac{1}{M}\sum_m \|\mathbf{t}_m - \bar{\mathbf{t}}\|^2}}{\sqrt{\frac{1}{M}\sum_m \|\mathbf{s}_m - \bar{\mathbf{s}}\|^2}}$$

3. **Rotation** via SVD of the cross-covariance:

$$\mathbf{H} = (s \cdot (\mathbf{S} - \bar{\mathbf{s}}))^\top (\mathbf{T} - \bar{\mathbf{t}}), \qquad \mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

$$\mathbf{R} = \mathbf{V}\,\mathrm{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^\top))\,\mathbf{U}^\top$$

where the $\det$ correction ensures $\mathbf{R} \in \mathrm{SO}(3)$.

4. **Translation**:

$$\mathbf{t} = \bar{\mathbf{t}} - s\,\mathbf{R}\,\bar{\mathbf{s}}$$

### 2.6 Transform Accumulation

Each pairwise alignment yields $\mathbf{A}_{k \leftarrow k+1} = (s_k, \mathbf{R}_k, \mathbf{t}_k)$ mapping chunk $k{+}1$ into chunk $k$'s frame. Cumulative transforms mapping chunk $k$ into chunk $0$'s (global) frame are composed as:

$$\mathbf{A}_{0 \leftarrow k+1} = \mathbf{A}_{0 \leftarrow k} \circ \mathbf{A}_{k \leftarrow k+1}$$

where $\mathrm{SIM}(3)$ composition is:

$$s_{0 \leftarrow k+1} = s_{0 \leftarrow k} \cdot s_{k \leftarrow k+1}$$

$$\mathbf{R}_{0 \leftarrow k+1} = \mathbf{R}_{0 \leftarrow k}\,\mathbf{R}_{k \leftarrow k+1}$$

$$\mathbf{t}_{0 \leftarrow k+1} = s_{0 \leftarrow k}\left(\mathbf{R}_{0 \leftarrow k}\,\mathbf{t}_{k \leftarrow k+1}\right) + \mathbf{t}_{0 \leftarrow k}$$

### 2.7 Global Pose Assembly

For frame $i$ in chunk $k > 0$, the globally-aligned camera-to-world pose is:

$$\mathbf{T}^{c2w, \text{global}}_i = \mathbf{A}_{0 \leftarrow k} \circ \left(\mathbf{T}^{(k)}_i\right)^{-1}$$

Concretely, the $\mathrm{SIM}(3)$ is applied as a $4 \times 4$ matrix:

$$\mathbf{S} = \begin{pmatrix} s\,\mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}, \qquad \tilde{\mathbf{T}}^{c2w} = \mathbf{S}\,\left(\mathbf{T}^{(k)}_i\right)^{-1}$$

The scale is then stripped from the rotation block to recover an $\mathrm{SE}(3)$ pose:

$$\mathbf{T}^{c2w, \text{global}}_i = \begin{pmatrix} \frac{1}{s}\,\tilde{\mathbf{T}}^{c2w}_{1:3, 1:3} & \tilde{\mathbf{T}}^{c2w}_{1:3, 4} \\ \mathbf{0}^\top & 1 \end{pmatrix}$$

Finally, $\mathbf{T}^{w2c, \text{global}}_i = \left(\mathbf{T}^{c2w, \text{global}}_i\right)^{-1}$.

Frames in the overlap region are assigned to the earlier chunk to avoid duplicates.

---

## 3. DA3 Pairwise with Depth-Based Scale Correction

### 3.1 Overview

The simplest online method: DA3 is run on each consecutive frame pair $(\mathcal{I}_{i-1}, \mathcal{I}_i)$, yielding a relative pose. Relative poses are chained multiplicatively into an absolute trajectory. Since DA3's per-pair depth predictions have arbitrary scale, a depth-ratio correction maintains metric consistency across pairs.

### 3.2 Pairwise Inference

DA3 batch inference on the pair $(\mathcal{I}_{i-1}, \mathcal{I}_i)$ returns:

- Extrinsics: $\mathbf{T}^{\text{pair}}_0 \approx \mathbf{I}_4$ (reference frame = first image), $\mathbf{T}^{\text{pair}}_1 \in \mathrm{SE}(3)$ (relative w2c)
- Depth maps: $D^{\text{pair}}_0, D^{\text{pair}}_1 \in \mathbb{R}^{H \times W}_+$
- Confidence: $c^{\text{pair}}_0, c^{\text{pair}}_1 \in \mathbb{R}^{H \times W}_+$

The relative transform is $\Delta\mathbf{T}_i = \mathbf{T}^{\text{pair}}_1$.

### 3.3 Depth-Based Scale Correction

Each pair's predictions live in an arbitrary metric scale. The shared frame $\mathcal{I}_{i-1}$ appears as the second image in pair $(i{-}2, i{-}1)$ and as the first image in pair $(i{-}1, i)$, yielding two depth predictions $D^{\text{old}}_{i-1}$ and $D^{\text{new}}_{i-1}$ of the same frame.

The scale correction factor is computed as the robust ratio of these depths over high-confidence pixels:

$$\mathcal{M} = \left\{(u,v) : c^{\text{old}}(u,v) > \tau \;\wedge\; c^{\text{new}}(u,v) > \tau \;\wedge\; D^{\text{old}}(u,v) > \epsilon \;\wedge\; D^{\text{new}}(u,v) > \epsilon\right\}$$

$$\alpha_i = \mathrm{median}_{(u,v) \in \mathcal{M}} \frac{D^{\text{old}}_{i-1}(u,v)}{D^{\text{new}}_{i-1}(u,v)}$$

where $\tau = 0.3 \cdot \min(\mathrm{median}(c^{\text{old}}), \mathrm{median}(c^{\text{new}}))$ and $\epsilon = 10^{-3}$.

The scale correction is applied to the translation component only:

$$\Delta\mathbf{T}_i[:3, 3] \leftarrow \alpha_i \cdot \Delta\mathbf{T}_i[:3, 3]$$

The rotation is left unchanged since DA3's rotation predictions are scale-invariant.

### 3.4 Pose Chaining

Absolute poses are computed by left-multiplying relative transforms:

$$\mathbf{T}^{w2c}_0 = \mathbf{I}_4, \qquad \mathbf{T}^{w2c}_i = \Delta\mathbf{T}_i \cdot \mathbf{T}^{w2c}_{i-1}$$

### 3.5 Drift Characteristics

Pairwise chaining accumulates errors multiplicatively. Without loop closure or global optimization:

- **Rotation drift**: $\mathcal{O}(\sqrt{N})$ under i.i.d. rotation noise (random walk on $\mathrm{SO}(3)$).
- **Translation drift**: $\mathcal{O}(N)$ in the worst case, mitigated partially by the depth-based scale correction which prevents scale divergence between consecutive pairs.
- **Scale drift**: The depth ratio $\alpha_i$ corrects *relative* scale between adjacent pairs but cannot prevent slow scale drift over long sequences, as errors in $\alpha_i$ accumulate multiplicatively.

This method trades accuracy for latency: each frame can be processed immediately upon arrival (no buffering), at the cost of the highest drift among the three online methods.

---

## 4. Comparison

| Property | StreamVGGT (windowed) | DA3 Chunked | DA3 Pairwise |
|---|---|---|---|
| Latency | 1 frame | $C$ frames (buffered) | 1 frame |
| Memory | $\mathcal{O}(W \cdot P \cdot L)$ | $\mathcal{O}(C^2)$ per chunk | $\mathcal{O}(1)$ |
| Drift | Bounded by window $W$ | Per-chunk: none; inter-chunk: $\mathrm{SIM}(3)$ error | $\mathcal{O}(N)$ |
| Scale | Metric (from training) | Per-chunk metric; cross-chunk via $\mathrm{SIM}(3)$ | Corrected via depth ratio |
| Coordinate frame | Consistent (camera head retains full history) | Chunk 0's local frame | First frame = identity |
| Loop closure | No | No (but overlap reduces inter-chunk drift) | No |

---

## 5. Implementation Notes

Our implementations are simplified variants of the upstream methods:

- **DA3 Chunked**: The upstream DA3-Streaming and VGGT-Long pipelines use **robust weighted $\mathrm{SIM}(3)$** with IRLS (iteratively reweighted least squares using Huber loss), confidence-weighted point correspondences, and optional loop closure via a $\mathrm{SIM}(3)$ pose graph optimizer. Our implementation uses the plain (unweighted, non-robust) Umeyama estimator and omits loop closure — sufficient for short sequences but would accumulate inter-chunk drift on long trajectories.

- **DA3 Pairwise**: This is our own simplified approach to frame-by-frame online inference. The depth-ratio scale correction is analogous to the scale precomputation in DA3-Streaming (which offers RANSAC and confidence-weighted variants), but applied per-pair rather than per-chunk.

---

## References

- **Umeyama, S.** (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE TPAMI*, 13(4), 376–380.
- **VGGT-Long** (Deng, K.): [github.com/DengKaiCQ/VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) — original chunked submap alignment with $\mathrm{SIM}(3)$ and loop closure for VGGT.
- **DA3-Streaming** (ByteDance): `Depth-Anything-3/da3_streaming/` — adapted from VGGT-Long for DA3, adding robust IRLS alignment, Triton/Numba acceleration, and confidence-weighted scale estimation.
- **Wang, J., et al.** (2025). Depth Anything 3. *arXiv preprint*.
