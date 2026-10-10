# ROADMAP.md — Future Architectural Directions for SINTEZ

This document outlines the theoretical developments and engineering milestones planned for **СИНТЕЗ (SINTEZ)** v1.1 through v2.0.

---

## 1. Differentiable Translation Operator & End-to-End Gradients (SINTEZ v1.5)

### Problem
In the current v1.0 architecture, the translation operator $`\mathcal{T}`$ is a discrete, non-differentiable mapping evaluated in Stage 2. Consequently, the DNA backbone can only be trained contrastively via counterfactual hard sampling, rather than through direct end-to-end gradient backpropagation from ESM-2.

### Proposed Architecture
- Formulate a **differentiable relaxation** of the genetic code lookup:

```math
\mathbf{e}_{\text{codon}} = \mathbf{x}_{i} \otimes \mathbf{x}_{i+1} \otimes \mathbf{x}_{i+2} \in \mathbb{R}^{64}
```

- Map the 64-dimensional codon simplex to amino acid continuous embeddings via a learned projection matrix $`\mathbf{W}_{\text{code}} \in \mathbb{R}^{20 \times 64}`$:

```math
\mathbf{z}_{\text{AA}} = \mathbf{W}_{\text{code}} \cdot \mathbf{e}_{\text{codon}}
```

- Route $`\mathbf{z}_{\text{AA}}`$ directly into the initial transformer layer of ESM-2, enabling backpropagation of protein folding stability gradients directly into the DNA splice junction convolutional filters.

---

## 2. ESM-2 Contact Map Junction Density (SINTEZ v1.6)

### Problem
Spurious chimeric gene models occasionally concatenate two authentic adjacent genes into a single giant transcript because both halves have low ESM perplexity.

### Proposed Solution
- Authentic multi-exon proteins fold into cohesive tertiary domains, exhibiting **dense residue-residue contacts across exon junctions**:

```math
\mathcal{C}_{\text{junction}} = \frac{1}{|E_1| \cdot |E_2|} \sum_{i \in E_1} \sum_{j \in E_2} P(\text{contact}(i, j) \mid \text{ESM-2})
```

- Chimeric concatenations will have a **contact desert** between the independent protein domains. Incorporating $`\mathcal{C}_{\text{junction}}`$ into the DAG edge scoring function will naturally penalize chimeric fusions without requiring artificial length ceilings.

---

## 3. Non-Canonical Splice Site Decoding (SINTEZ v1.7)

### Motivation
While >98% of eukaryotic introns follow canonical `GT...AG` rules, non-canonical introns (`GC...AG` and U12-type `AT...AC`) account for critical regulatory and highly conserved genes in complex metazoans.

### Implementation
- Dynamically expand the acceptor/donor search dictionary in `sintez.dag_decoder` to include `GC` donors and `AT...AC` pairs when:

```math
P_{\text{donor}}(d) \times P_{\text{acceptor}}(a) \gt \tau_{\text{noncanonical}}
```

- Implement an explicit Bayesian prior penalty $`\lambda_{\text{noncanon}} \approx 3.0`$ nats to reflect the empirical 1:100 biological frequency ratio.

---

## 4. Multi-GPU Distributed Tile Streaming for Mammalian Chromosomes (SINTEZ v2.0)

### Motivation
Mammalian chromosomes (e.g., Human Chromosome 1: 248.95 Mb) require gigabyte-scale memory when tiled at high overlap.

### Implementation
- Introduce an asynchronous tile streaming pipeline with PyTorch Distributed Data Parallel (`torch.distributed`) or Ray:
  - Worker pool partitions the chromosome into 50 kb windows with 10 kb halos.
  - Workers compute local DNA backbone tensors in parallel.
  - Master node executes streaming topological DAG stitching across halo boundaries.

---

## 5. Integration with Empirical Evolutionary Rate Profiles (HyPhy / PRIME)

### Motivation
By coupling structural protein language models with empirical evolutionary conservation profiles:
- Sites identified as under intense purifying selection or property conservation (via HyPhy/PRIME) provide an independent orthogonal constraint on coding vs non-coding nature.
- Highly conserved codon sites act as anchors for DAG path decoders, boosting sensitivity in deep phylogenetic lineages.
