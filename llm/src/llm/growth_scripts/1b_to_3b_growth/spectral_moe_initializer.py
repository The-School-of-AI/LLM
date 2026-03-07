# spectral_moe_initializer.py

import torch
import torch.nn as nn


class SpectralMoEInitializer:
    """
    Initializes a 3B MoE model from a trained 1B dense model
    using SVD-based SwiGLU compression + structured rotation noise.
    """

    def __init__(
        self,
        dense_model: nn.Module,
        moe_model: nn.Module,
        num_routed_experts: int = 20,
        intermediate_dense: int = 2048,
        intermediate_moe: int = 1024,
        rotation_eps: float = 0.02,
        device: str = "cuda",
        svd_mode: str = "joint",
    ):
        assert svd_mode in ("joint", "independent"), \
            f"svd_mode must be 'joint' or 'independent', got '{svd_mode}'"
        self.dense_model = dense_model
        self.moe_model = moe_model
        self.num_routed_experts = num_routed_experts
        self.intermediate_dense = intermediate_dense
        self.intermediate_moe = intermediate_moe
        self.rotation_eps = rotation_eps
        self.device = device
        self.svd_mode = svd_mode

    # ---------------------------------------------------------
    # Core SVD Compression
    # ---------------------------------------------------------

    def _compress_swiglu(self, Wg, Wu, Wd):
        """
        Dispatch to joint or independent SVD based on self.svd_mode.
        """
        if self.svd_mode == "joint":
            return self._compress_swiglu_joint(Wg, Wu, Wd)
        else:
            return self._compress_swiglu_independent(Wg, Wu, Wd)

    def _compress_swiglu_joint(self, Wg, Wu, Wd):
        """
        Compress intermediate dimension (e.g. 1024 -> 512) using Joint SVD.
        Finds the subspace that best preserves Gate, Up, AND Down projections.

        Wg: (int_dense, hidden)  [(1024, 512)]
        Wu: (int_dense, hidden)  [(1024, 512)]
        Wd: (hidden, int_dense)  [(512, 1024)]
        """
        # We want to reduce intermediate dim from 1024 to 512.
        k = self.intermediate_moe

        # 1. Construct Joint Matrix M
        # We want to find important directions in the INTERMEDIATE space (dim 1024).
        # Wg.T: (512, 1024) - maps hidden -> intermediate
        # Wu.T: (512, 1024) - maps hidden -> intermediate
        # Wd:   (512, 1024) - maps intermediate -> hidden

        # Stack them all: M shape (3*512, 1024) = (1536, 1024)
        M = torch.cat([Wg.T, Wu.T, Wd], dim=0)

        # 2. Perform SVD on Joint Matrix
        # M = U S Vh
        # Vh rows are the principal directions in the shared intermediate space.
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Calculate Explained Variance
        explained_variance = (S[:k] ** 2).sum() / (S ** 2).sum()
        print(f"   📊 Joint SVD Compression ({Vh.shape[0]}->{k}): Explained Variance = {explained_variance.item():.4f}")

        if explained_variance < 0.90:
            print(f"   ⚠️  WARNING: Aggressive compression! Retained only {explained_variance.item()*100:.1f}% of signal.")

        # 3. Take top k components
        # Vh is (min(3*h, int), int) -> (1024, 1024)
        if Vh.shape[0] > k:
            V_k = Vh[:k, :] # (k, 1024)
        else:
            V_k = Vh

        # 4. Project weights into this shared subspace

        # Wg is (1024, 512). V_k is (k, 1024).
        # Wg_new = V_k @ Wg -> (k, 512).

        # Wd_new = Wd @ V_k.T
        # Shape: (512, 1024) @ (1024, k) -> (512, k)
        Wd_base = Wd @ V_k.T

        # Wg_new = V_k @ Wg
        # Shape: (k, 1024) @ (1024, 512) -> (k, 512)
        Wg_base = V_k @ Wg

        # Wu_new = V_k @ Wu
        # Shape: (k, 1024) @ (1024, 512) -> (k, 512)
        Wu_base = V_k @ Wu

        return Wg_base, Wu_base, Wd_base

    def _compress_swiglu_independent(self, Wg, Wu, Wd):
        """
        Compress intermediate dimension using per-matrix SVD with consensus basis.

        Unlike joint SVD (which stacks [Wg.T, Wu.T, Wd] into one matrix), this
        lets each matrix independently nominate its best k directions in the
        intermediate space, then forms a consensus basis via SVD of the
        concatenated candidates.

        Why not fully independent?  SwiGLU computes silu(x@Wg.T) * (x@Wu.T).
        The element-wise multiply requires gate and up activations to share the
        SAME coordinate system.  Fully separate SVD bases would make dimension i
        in gate point in a totally different direction than dimension i in up,
        turning the multiply into noise (hence the loss spike 4 -> 11).

        Ablation value:  Joint SVD lets all 3 matrices vote equally on the basis.
        This method lets each matrix independently find its best directions, then
        builds a shared basis from those candidates.  If one matrix dominates the
        joint SVD spectrum, independent nomination + consensus can surface
        directions that joint SVD would miss.

        Wg: (int_dense, hidden)  [(1024, 512)]
        Wu: (int_dense, hidden)  [(1024, 512)]
        Wd: (hidden, int_dense)  [(512, 1024)]
        """
        k = self.intermediate_moe
        int_dense = Wg.shape[0]

        # --- Step 1: Per-matrix SVD to find each matrix's best intermediate directions ---

        # Wg: (int_dense, hidden) → left singular vectors U_g span best subspace in intermediate dim
        U_g, S_g, _ = torch.linalg.svd(Wg, full_matrices=False)
        ev_g = (S_g[:k] ** 2).sum() / (S_g ** 2).sum()
        print(f"   📊 Independent SVD [Wg] ({int_dense}->{k}): "
              f"Explained Variance = {ev_g.item():.4f}")

        # Wu: (int_dense, hidden) → left singular vectors U_u
        U_u, S_u, _ = torch.linalg.svd(Wu, full_matrices=False)
        ev_u = (S_u[:k] ** 2).sum() / (S_u ** 2).sum()
        print(f"   📊 Independent SVD [Wu] ({int_dense}->{k}): "
              f"Explained Variance = {ev_u.item():.4f}")

        # Wd: (hidden, int_dense) → right singular vectors Vh_d span best subspace
        #   equivalently, left singular vectors of Wd.T
        _, S_d, Vh_d = torch.linalg.svd(Wd, full_matrices=False)
        ev_d = (S_d[:k] ** 2).sum() / (S_d ** 2).sum()
        print(f"   📊 Independent SVD [Wd] ({int_dense}->{k}): "
              f"Explained Variance = {ev_d.item():.4f}")

        # --- Step 2: Form consensus basis from candidates ---
        # Each matrix nominates its top-k directions in intermediate space (int_dense).
        # U_g[:,:k]:  (int_dense, k)  — Wg's best directions
        # U_u[:,:k]:  (int_dense, k)  — Wu's best directions
        # Vh_d[:k,:].T: (int_dense, k)  — Wd's best directions
        # Concatenate → (int_dense, 3k), then SVD to find top-k consensus directions.

        candidates = torch.cat([
            U_g[:, :k],       # (int_dense, k)
            U_u[:, :k],       # (int_dense, k)
            Vh_d[:k, :].T,    # (int_dense, k)
        ], dim=1)  # (int_dense, 3k)

        U_consensus, _, _ = torch.linalg.svd(candidates, full_matrices=False)
        V_k = U_consensus[:, :k].T  # (k, int_dense) — shared consensus basis

        # --- Step 3: Project all weights into consensus basis (same as joint) ---
        Wg_base = V_k @ Wg       # (k, hidden)
        Wu_base = V_k @ Wu       # (k, hidden)
        Wd_base = Wd @ V_k.T     # (hidden, k)

        # Report consensus explained variance (how much signal the consensus basis captures)
        # Measure per-matrix: ||V_k @ W||_F / ||W||_F for Wg/Wu, ||W @ V_k.T||_F / ||W||_F for Wd
        ev_consensus_g = (Wg_base.norm() ** 2) / (Wg.norm() ** 2)
        ev_consensus_u = (Wu_base.norm() ** 2) / (Wu.norm() ** 2)
        ev_consensus_d = (Wd_base.norm() ** 2) / (Wd.norm() ** 2)
        print(f"   📊 Consensus basis retained energy: "
              f"Wg={ev_consensus_g.item():.4f}, Wu={ev_consensus_u.item():.4f}, Wd={ev_consensus_d.item():.4f}")

        return Wg_base, Wu_base, Wd_base

    # ---------------------------------------------------------
    # Structured Rotation Noise
    # ---------------------------------------------------------

    def _random_small_rotation(self, dim):
        A = torch.randn(dim, dim, device=self.device)
        A = A - A.T  # skew-symmetric
        R = torch.matrix_exp(self.rotation_eps * A)
        return R

    # ---------------------------------------------------------
    # Build Routed Experts
    # ---------------------------------------------------------

    def _build_routed_experts(self, Wg, Wu, Wd):
        Wg_base, Wu_base, Wd_base = self._compress_swiglu(Wg, Wu, Wd)
        # Shapes after compression:
        #   Wg_base: (k, hidden)  — gate_proj weight convention
        #   Wu_base: (k, hidden)  — up_proj weight convention
        #   Wd_base: (hidden, k)  — down_proj weight convention
        # R rotates in the k-dimensional intermediate space.

        experts = []

        for _ in range(self.num_routed_experts):
            R = self._random_small_rotation(self.intermediate_moe)

            # Rotate in intermediate (k) space:
            #   gate/up: R @ W  rotates intermediate-dim rows  (k,k) @ (k,h) = (k,h)
            #   down:    W @ R.T rotates intermediate-dim cols  (h,k) @ (k,k) = (h,k)
            # This preserves SwiGLU function: silu(x@Wg.T@R.T) * (x@Wu.T@R.T) @ R@Wd.T
            # For small eps, rotations approximately cancel → expert(x) ≈ base(x)
            Wg_i = R @ Wg_base          # (k, hidden)
            Wu_i = R @ Wu_base          # (k, hidden)
            Wd_i = Wd_base @ R.T        # (hidden, k)
            
            # FIX: Scale down routed experts to prevent "Double Counting" at init.
            # Shared Expert already has 100% of 1B weights.
            # Routed experts should start as small residuals to avoid loss spike (4 -> 12).
            # Wd_i = Wd_i * 0.01  <-- REMOVED per user request 

            experts.append((Wg_i, Wu_i, Wd_i))

        return experts

    # ---------------------------------------------------------
    # Copy Shared Expert (Full 2048)
    # ---------------------------------------------------------

    def _copy_shared_expert(self, dense_mlp, moe_shared_mlp):
        moe_shared_mlp.gate_proj.weight.data.copy_(
            dense_mlp.gate_proj.weight.data
        )
        moe_shared_mlp.up_proj.weight.data.copy_(
            dense_mlp.up_proj.weight.data
        )
        moe_shared_mlp.down_proj.weight.data.copy_(
            dense_mlp.down_proj.weight.data
        )

    # ---------------------------------------------------------
    # Initialize Full Model
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Dense Weight Extraction Helpers
    # ---------------------------------------------------------

    def _extract_dense_weights(self, dense_lightning_mlp):
        """
        Extract (Wg, Wu, Wd) from a 1B dense LightningMLP.

        The 1B LightningMLP wraps DenseMLP which wraps LigerSwiGLUMLP:
          LightningMLP.mlp -> DenseMLP.mlp -> LigerSwiGLUMLP (gate_proj, up_proj, down_proj)

        Returns gate, up, down weight tensors on self.device.
        """
        liger_mlp = dense_lightning_mlp.mlp.mlp  # DenseMLP -> LigerSwiGLUMLP
        Wg = liger_mlp.gate_proj.weight.data.to(self.device)
        Wu = liger_mlp.up_proj.weight.data.to(self.device)
        Wd = liger_mlp.down_proj.weight.data.to(self.device)
        return Wg, Wu, Wd

    def initialize(self):
        """
        Refined for Model3B: Direct mapping to batched expert tensors.
        Also copies all non-MoE components (Embeddings, Attention, Norms, Head).

        The 1B dense model uses LightningMLP(DenseMLP(LigerSwiGLUMLP)) with attributes:
          .mlp.mlp.{gate_proj, up_proj, down_proj}
        The 3B MoE model uses LightningMLP(MoEFFN) with attributes:
          .moe.{shared_gate, shared_up, shared_down, W_gate, W_up, W_down}
        """
        print("   Copying non-MoE components (Embeddings, Attention, Head)...")

        # 1. Copy Embeddings
        if hasattr(self.dense_model, 'kronecker_embeddings') and hasattr(self.moe_model, 'kronecker_embeddings'):
            self.moe_model.kronecker_embeddings.load_state_dict(self.dense_model.kronecker_embeddings.state_dict())
        elif hasattr(self.dense_model, 'token_embed') and hasattr(self.moe_model, 'token_embed'):
            self.moe_model.token_embed.load_state_dict(self.dense_model.token_embed.state_dict())

        # Copy Embedding Norm (critical for equivalence)
        if hasattr(self.dense_model, 'embed_norm') and hasattr(self.moe_model, 'embed_norm'):
            if self.dense_model.embed_norm is not None and self.moe_model.embed_norm is not None:
                self.moe_model.embed_norm.load_state_dict(self.dense_model.embed_norm.state_dict())

        # 2. Copy Final Norm and Head
        self.moe_model.norm.load_state_dict(self.dense_model.norm.state_dict())
        self.moe_model.lm_head.load_state_dict(self.dense_model.lm_head.state_dict())

        # 3. Copy other components (Kronecker projection, Memory Gate)
        if hasattr(self.dense_model, 'pf_to_model') and hasattr(self.moe_model, 'pf_to_model'):
            if self.dense_model.pf_to_model is not None and self.moe_model.pf_to_model is not None:
                self.moe_model.pf_to_model.load_state_dict(self.dense_model.pf_to_model.state_dict())

        if hasattr(self.dense_model, 'memory_gate_proj') and hasattr(self.moe_model, 'memory_gate_proj'):
             self.moe_model.memory_gate_proj.load_state_dict(self.dense_model.memory_gate_proj.state_dict())

        # MTP Block Initialization (Dense -> MoE)
        # The MTP block in 3B model uses MoE LightningMLP, while in 1B it uses Dense LightningMLP.
        # We must initialize it using SVD/Spectral init, just like the main layers.
        if hasattr(self.dense_model, 'mtp_block') and hasattr(self.moe_model, 'mtp_block'):
            if self.dense_model.mtp_block is not None and self.moe_model.mtp_block is not None:
                print("   Initializing MTP Block (Dense -> MoE)...")
                dense_mtp = self.dense_model.mtp_block
                moe_mtp = self.moe_model.mtp_block

                # 1. Copy Fusion projection
                moe_mtp.fusion_proj.load_state_dict(dense_mtp.fusion_proj.state_dict())

                # 2. Copy Attention module
                moe_mtp.attn.load_state_dict(dense_mtp.attn.state_dict())

                # 3. Copy MHC wrapper norms/coeffs for attn_block and mlp_block.
                # In 3B with mtp_reversible=True, attn_block/mlp_block are set via
                # object.__setattr__ (not registered submodules) but still accessible.
                moe_mtp.attn_block.coeffs.load_state_dict(dense_mtp.attn_block.coeffs.state_dict())
                moe_mtp.attn_block.norm.load_state_dict(dense_mtp.attn_block.norm.state_dict())
                moe_mtp.mlp_block.coeffs.load_state_dict(dense_mtp.mlp_block.coeffs.state_dict())
                moe_mtp.mlp_block.norm.load_state_dict(dense_mtp.mlp_block.norm.state_dict())

                # 4. Initialize MoE MLP from Dense MLP weights
                # 1B: LightningMLP.mlp -> DenseMLP.mlp -> LigerSwiGLUMLP
                # 3B: LightningMLP.moe -> MoEFFN (shared_gate/up/down + W_gate/up/down)
                Wg, Wu, Wd = self._extract_dense_weights(dense_mtp.mlp)
                moe_layer = moe_mtp.mlp.moe  # Access inner MoEFFN

                # Copy Shared Expert
                moe_layer.shared_gate.weight.data.copy_(Wg)
                moe_layer.shared_up.weight.data.copy_(Wu)
                moe_layer.shared_down.weight.data.copy_(Wd)

                # Build & Assign Routed Experts
                routed_weights = self._build_routed_experts(Wg, Wu, Wd)

                if hasattr(moe_layer, 'W_gate') and isinstance(moe_layer.W_gate, nn.Parameter):
                    # Batched MTP Experts
                    Wg_stack = []
                    Wu_stack = []
                    Wd_stack = []
                    for idx in range(self.num_routed_experts):
                        Wg_i, Wu_i, Wd_i = routed_weights[idx]
                        Wg_stack.append(Wg_i.T)
                        Wu_stack.append(Wu_i.T)
                        Wd_stack.append(Wd_i.T)

                    moe_layer.W_gate.data.copy_(torch.stack(Wg_stack, dim=0))
                    moe_layer.W_up.data.copy_(torch.stack(Wu_stack, dim=0))
                    moe_layer.W_down.data.copy_(torch.stack(Wd_stack, dim=0))

                elif hasattr(moe_layer, 'routed_experts'):
                    # Module List MTP Experts
                    for idx, expert in enumerate(moe_layer.routed_experts):
                        Wg_i, Wu_i, Wd_i = routed_weights[idx]
                        expert.gate_proj.weight.data.copy_(Wg_i)
                        expert.up_proj.weight.data.copy_(Wu_i)
                        expert.down_proj.weight.data.copy_(Wd_i)

                # Bias MTP router: moderate null preference at init
                if hasattr(moe_layer, 'gate') and moe_layer.gate is not None:
                    if hasattr(moe_layer.gate, 'logit_bias'):
                        nn.init.constant_(moe_layer.gate.logit_bias, 0.0)
                    if hasattr(moe_layer.gate, 'null_logit'):
                        nn.init.constant_(moe_layer.gate.null_logit, 0.0)

        print("   Copying Layer weights (Attention + MLPs)...")
        for layer_idx in range(len(self.dense_model.layers)):
            dense_block = self.dense_model.layers[layer_idx]
            moe_block = self.moe_model.layers[layer_idx]

            # 4. Copy Attention Block (Entire Sublayer: Coeffs + Norm + Attention)
            moe_block.attn_block.load_state_dict(dense_block.attn_block.state_dict())

            # 5. Copy MLP Block Wrapper (Coeffs + Norm ONLY)
            # We CANNOT copy the entire mlp_block because sublayer is different (MLP vs MoE)
            moe_block.mlp_block.coeffs.load_state_dict(dense_block.mlp_block.coeffs.state_dict())
            moe_block.mlp_block.norm.load_state_dict(dense_block.mlp_block.norm.state_dict())

            # Extract dense weights from 1B LightningMLP(DenseMLP(LigerSwiGLUMLP))
            dense_lightning_mlp = dense_block.mlp_block.sublayer  # LightningMLP (1B)
            Wg, Wu, Wd = self._extract_dense_weights(dense_lightning_mlp)

            # Access 3B MoEFFN
            moe_layer = moe_block.mlp_block.sublayer.moe

            # 1️⃣ Shared expert copy (Standard 2048 size)
            moe_layer.shared_gate.weight.data.copy_(Wg)
            moe_layer.shared_up.weight.data.copy_(Wu)
            moe_layer.shared_down.weight.data.copy_(Wd)

            # 2️⃣ Routed experts
            # Remove scaling! We want full gradients.
            # We rely on Router Bias to keep them silent at init.
            routed_weights = self._build_routed_experts(Wg, Wu, Wd)
            
            # Check for batched implementation (MoEFFN in recurrence_model_3b.py)
            if hasattr(moe_layer, 'W_gate') and isinstance(moe_layer.W_gate, nn.Parameter):
                # Batched: (num_experts, d_model, d_hidden)
                # routed_weights is list of (Wg, Wu, Wd) where Wg is (d_model, d_hidden)
                
                # Stack weights
                Wg_stack = []
                Wu_stack = []
                Wd_stack = []
                
                for idx in range(self.num_routed_experts):
                    Wg_i, Wu_i, Wd_i = routed_weights[idx]
                    
                    # Store
                    Wg_stack.append(Wg_i.T) # (d_hidden, d_model).T -> (d_model, d_hidden)
                    Wu_stack.append(Wu_i.T) # (d_hidden, d_model).T -> (d_model, d_hidden)
                    Wd_stack.append(Wd_i.T) # (d_model, d_hidden).T -> (d_hidden, d_model)
                    
                # Stack to (num_experts, ...)
                moe_layer.W_gate.data.copy_(torch.stack(Wg_stack, dim=0))
                moe_layer.W_up.data.copy_(torch.stack(Wu_stack, dim=0))
                moe_layer.W_down.data.copy_(torch.stack(Wd_stack, dim=0))
                
            # Check for list implementation (Generic MoE)
            elif hasattr(moe_layer, 'routed_experts'): 
                for idx, expert in enumerate(moe_layer.routed_experts):
                    Wg_i, Wu_i, Wd_i = routed_weights[idx]

                    expert.gate_proj.weight.data.copy_(Wg_i)
                    expert.up_proj.weight.data.copy_(Wu_i)
                    expert.down_proj.weight.data.copy_(Wd_i)

            # 3️⃣ Bias Router to Null Experts
            # Real experts (-5) vs Null experts (+5).
            # Softmax will pick Null experts (output 0). 
            # Loss = Shared + 0 approx Dense.
            # But Real experts have FULL strength gradients when selected.
            if hasattr(moe_layer, 'gate') and moe_layer.gate is not None:
                # FIX: Removing gate zeroing to allow diversity at init
                # if hasattr(moe_layer.gate, 'gate') and isinstance(moe_layer.gate.gate, nn.Linear):
                #      nn.init.zeros_(moe_layer.gate.gate.weight)
                     
                if hasattr(moe_layer.gate, 'logit_bias'):
                    nn.init.constant_(moe_layer.gate.logit_bias, 0.0)
                if hasattr(moe_layer.gate, 'null_logit'):
                    nn.init.constant_(moe_layer.gate.null_logit, 0.0)  # Gap=2.0, moderate null preference
            else:
                print(f"⚠️ Warning: Could not identify expert structure in layer {layer_idx}")

        print(f"MoE model initialized from dense model using spectral compression (svd_mode='{self.svd_mode}').")

        print(f"✅ Successfully initialized {len(self.moe_model.layers)} layers.")

    def pairwise_similarity(experts):
        flats = [e["gate"].flatten() for e in experts]
        sims = []
        for i in range(len(flats)):
            for j in range(i+1, len(flats)):
                sims.append(torch.cosine_similarity(flats[i], flats[j], dim=0))
        return torch.stack(sims).mean()

    def validate_expert_diversity(self):
        """
        Analyzes the similarity between experts to ensure symmetry breaking.
        Target: 0.95 - 0.98 mean similarity.
        """
        print("\n📊 --- Expert Diversity Report ---")
    
        for layer_idx, layer in enumerate(self.moe_model.layers):
            # Access batched tensors from Model3B MoEFFN
            moe_layer = layer.mlp_block.sublayer.moe
            
            # Flatten each expert's gate weight for comparison
            # Shape: [num_experts, d_model * d_hidden]
            expert_weights = moe_layer.W_gate.data.view(self.num_routed_experts, -1)
            
            # Compute pairwise cosine similarity matrix
            # (normalized weights) @ (normalized weights).T
            norm = expert_weights.norm(p=2, dim=1, keepdim=True)
            normalized_weights = expert_weights / (norm + 1e-8)
            sim_matrix = torch.mm(normalized_weights, normalized_weights.t())
            
            # Extract upper triangle (excluding diagonal) to get unique pairs
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            unique_sims = sim_matrix[mask]
            
            mean_sim = unique_sims.mean().item()
            min_sim = unique_sims.min().item()
            max_sim = unique_sims.max().item()
            
            status = "✅ OPTIMAL" if 0.95 <= mean_sim <= 0.98 else "⚠️  ADJUST EPS"
            
            print(f"Layer {layer_idx:02d} | Mean Sim: {mean_sim:.4f} | Range: [{min_sim:.3f}, {max_sim:.3f}] | {status}")

