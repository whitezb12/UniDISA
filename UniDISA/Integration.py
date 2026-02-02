import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *
from itertools import chain
import os


class IntegrationModel:
    def __init__(
        self,
        adata_A: AnnData,
        adata_B: AnnData,
        input_key: List[str] = ["X_pca", "X_lsi"],
        batch_size: int = 500,
        n_latent: int = 10,
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed: int = 1234,
    ):

        self.device = device

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.celltype_col = celltype_col
        self.source_col = source_col
        
        celltypes_A = np.unique(adata_A.obs[self.celltype_col].dropna()) if self.celltype_col in adata_A.obs else np.array([])
        celltypes_B = np.unique(adata_B.obs[self.celltype_col].dropna()) if self.celltype_col in adata_B.obs else np.array([])
        self.unique_celltypes = np.union1d(celltypes_A, celltypes_B)

        self.dataset_A = AnnDataDataset(
            adata_A,
            input_key=input_key[0],
            output_layer=None,
            celltype_key=self.celltype_col,
            source_key=self.source_col,
            mode="integration",
            unique_labels=self.unique_celltypes
        )
        self.dataset_B = AnnDataDataset(
            adata_B,
            input_key=input_key[1],
            output_layer=None,
            celltype_key=self.celltype_col,
            source_key=self.source_col,
            mode="integration",
            unique_labels=self.unique_celltypes
        )

        self.dataloader_A = load_data(self.dataset_A, batch_size=self.batch_size, mode="integration")
        self.dataloader_B = load_data(self.dataset_B, batch_size=self.batch_size, mode="integration")

        self.is_shared_A = torch.ones(adata_A.shape[0], dtype=torch.bool, device=self.device)
        self.is_shared_B = torch.ones(adata_B.shape[0], dtype=torch.bool, device=self.device)


    def train_stage1(
        self,
        training_steps: int = 2000,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 1.0,
        lambdaDA: float = 1.0,
    ):

        self._init_models_and_optimizers()
        self._init_low_encoders()
        self._set_train_mode()

        print("===== Stage 1: Initialize Matching =====")

        iterator_A = iter(self.dataloader_A)
        iterator_B = iter(self.dataloader_B)

        for step in range(training_steps + 1):

            batch_A = next(iterator_A)
            batch_B = next(iterator_B)

            x_A = batch_A["input"].float().to(self.device)
            x_B = batch_B["input"].float().to(self.device)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            x_Arec = self.G_A(z_A)
            x_Brec = self.G_B(z_B)                
            
            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arec - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brec - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA 

            # optimal transport process
            C = pairwise_correlation_distance(batch_A["link_feat"], batch_B["link_feat"]).to(self.device)
            P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)

            # distribution alignment 
            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_DA = torch.sum(P * z_dist) / torch.sum(P)

            total_loss = (
                lambdaRecon * loss_AE
                + lambdaLA * loss_LA
                + lambdaDA * loss_DA
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 500 == 0:
                print(
                    f"[Stage1 {step}] "
                    f"AE: {loss_AE.item():.4f} | "
                    f"LA: {loss_LA.item():.4f} | "
                    f"DA: {loss_DA.item():.4f}"
                )

        self.update_slow_encoder()


    def train_stage2(
        self,
        training_steps: int = 3000,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 1.0,
        lambdaDA: float = 1.0,
        iters: int = 1,
    ):

        self._init_models_and_optimizers()
        self._set_train_mode()

        for it in range(1, iters+1):
            print(f"===== Stage 2: Iterative Alignment {it}/{iters} =====")

            iterator_A = iter(self.dataloader_A)
            iterator_B = iter(self.dataloader_B)

            for step in range(training_steps + 1):

                batch_A = next(iterator_A)
                batch_B = next(iterator_B)

                x_A = batch_A["input"].float().to(self.device)
                x_B = batch_B["input"].float().to(self.device)

                _, link_A, _ = self.E_A_slow(x_A)
                _, link_B, _ = self.E_B_slow(x_B)

                z_A, mu_A, logvar_A = self.E_A_fast(x_A)
                z_B, mu_B, logvar_B = self.E_B_fast(x_B)

                x_Arec = self.G_A(z_A)
                x_Brec = self.G_B(z_B)                
                
                _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
                _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

                # input autoencoder loss
                beta = 0.01
                loss_AE_A = torch.mean((x_Arec - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
                loss_AE_B = torch.mean((x_Brec - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
                loss_AE = loss_AE_A + loss_AE_B

                # latent align loss
                loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
                loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
                loss_LA = loss_LA_AtoB + loss_LA_BtoA 

                # optimal transport process
                C = pairwise_correlation_distance(link_A, link_B).to(self.device)
                P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)

                # distribution alignment 
                sigma_A = torch.exp(0.5 * logvar_A)
                sigma_B = torch.exp(0.5 * logvar_B)
                z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
                loss_DA = torch.sum(P * z_dist) / torch.sum(P)

                total_loss = (
                    lambdaRecon * loss_AE
                    + lambdaLA * loss_LA
                    + lambdaDA * loss_DA
                )

                self.optimizer_G.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
                self.optimizer_G.step()

                if step % 500 == 0:
                    print(
                        f"[Stage2 {step}] "
                        f"AE: {loss_AE.item():.4f} | "
                        f"LA: {loss_LA.item():.4f} | "
                        f"DA: {loss_DA.item():.4f}"
                    )
            if it < iters:
                self.update_slow_encoder()


    def train_stage3(
        self,
        training_steps: int = 10000,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 1.0,
        lambdaDA: float = 1.0,
        lambdamGAN: float = 0.5,
        lambdabGAN: float = 1.0,
        lambdaSemi: float=1.0
    ):

        self._init_models_and_optimizers()
        if self.celltype_col is not None:
            self.CLS = Classfier(self.n_latent, self.unique_celltypes.shape[0])
            self.CLS = self.CLS.to(self.device)
            self.params_G += list(self.CLS.parameters()) 
        self._set_train_mode()

        print("===== Stage 3: Shared Alignment =====")

        iterator_A = iter(self.dataloader_A)
        iterator_B = iter(self.dataloader_B)

        for step in range(training_steps + 1):

            batch_A = next(iterator_A)
            batch_B = next(iterator_B)

            mask_A = self.is_shared_A[batch_A["index"]]
            mask_B = self.is_shared_B[batch_B["index"]]
            if mask_A.sum() < 100 or mask_B.sum() < 100:
                continue

            x_A = batch_A["input"].float().to(self.device)
            x_B = batch_B["input"].float().to(self.device)

            _, link_A, _ = self.E_A_slow(x_A)
            _, link_B, _ = self.E_B_slow(x_B)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            x_Arec = self.G_A(z_A)
            x_Brec = self.G_B(z_B)                
            
            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arec - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brec - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA 

            # optimal transport process
            C = pairwise_correlation_distance(link_A, link_B).to(self.device)
            P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)

            # distribution alignment 
            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_DA = torch.sum(P * z_dist) / torch.sum(P)

            # discriminator loss
            for _ in range(5):
                self.optimizer_Dis_m.zero_grad() 
                loss_mDis_A = (F.softplus(-self.Dis_Z(z_A[mask_A].detach()))).mean()
                loss_mDis_B = (F.softplus(self.Dis_Z(z_B[mask_B].detach()))).mean()
                loss_mDis = loss_mDis_A + loss_mDis_B
                loss_mDis.backward()
                self.optimizer_Dis_m.step()

            if self.optimizer_Dis_b:
                self.optimizer_Dis_b.zero_grad()
                loss_bDis = self.compute_discriminator_loss_intra(z_A.detach(), z_B.detach(), batch_A['source'], batch_B['source'])
                loss_bDis.backward()
                self.optimizer_Dis_b.step()
            else:
                loss_bDis = torch.tensor(0.0, device=self.device)

            # generator loss
            loss_mGAN_A = -(F.softplus(-self.Dis_Z(z_A[mask_A]))).mean()
            loss_mGAN_B = -(F.softplus(self.Dis_Z(z_B[mask_B]))).mean()
            loss_mGAN = loss_mGAN_A + loss_mGAN_B
            loss_bGAN = -self.compute_discriminator_loss_intra(z_A, z_B, batch_A['source'], batch_B['source'])   

            # semi loss
            if self.celltype_col is not None:
                loss_semi = self._compute_semi_loss(z_A, z_B, batch_A['celltype'], batch_B['celltype'])
            else:
                loss_semi = torch.tensor(0.0, device=z_A.device)
            
            total_loss = (
                lambdaRecon * loss_AE
                + lambdaLA * loss_LA
                + lambdaDA * loss_DA
                + lambdamGAN * loss_mGAN
                + lambdabGAN * loss_bGAN
                + lambdaSemi * loss_semi
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                print(
                    f"[Stage3 {step}] "
                    f"AE: {loss_AE:.4f} | "
                    f"LA: {loss_LA:.4f} | "
                    f"DA: {loss_DA:.4f} | "
                    f"mGAN: {loss_mGAN:.4f} | "
                    f"bGAN: {loss_bGAN:.4f} | "
                    f"Semi: {loss_semi:.4f}"
                )


    def compute_discriminator_loss_intra(self, z_A, z_B, source_A, source_B):
        losses = []
        if self.Dis_A:
            losses.append(F.cross_entropy(self.Dis_A(z_A), source_A.to(self.device)))
        if self.Dis_B:
            losses.append(F.cross_entropy(self.Dis_B(z_B), source_B.to(self.device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return sum(losses)
    

    def _compute_semi_loss(self, z_A: torch.Tensor, z_B: torch.Tensor, celltype_A: torch.Tensor, celltype_B: torch.Tensor, label_smoothing: float = 0.1) -> torch.Tensor:
        loss_semi = torch.tensor(0.0, device=z_A.device)
        
        mask_cls_A = (celltype_A != -1)
        mask_cls_B = (celltype_B != -1)

        if mask_cls_A.sum() == 0 and mask_cls_B.sum() == 0:
            return loss_semi
        
        logits_A = self.CLS(z_A)
        logits_B = self.CLS(z_B)

        def label_smoothed_nll_loss(lprobs, target, eps):
            nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = -lprobs.mean(dim=-1)
            loss = (1.0 - eps) * nll_loss + eps * smooth_loss
            return loss.sum()

        loss_semi_A = 0.0
        if mask_cls_A.sum() > 0:
            lprobs_A = F.log_softmax(logits_A[mask_cls_A], dim=-1)
            loss_semi_A = label_smoothed_nll_loss(lprobs_A, celltype_A[mask_cls_A].to(self.device), label_smoothing)
        
        loss_semi_B = 0.0
        if mask_cls_B.sum() > 0:
            lprobs_B = F.log_softmax(logits_B[mask_cls_B], dim=-1)
            loss_semi_B = label_smoothed_nll_loss(lprobs_B, celltype_B[mask_cls_B].to(self.device), label_smoothing)
    
        loss_semi = loss_semi_A + loss_semi_B
        return loss_semi
    

    def _init_models_and_optimizers(self):
        self.E_A_fast = VAEEncoder(self.dataset_A.feature_shapes["input"], self.n_latent).to(self.device)
        self.E_B_fast = VAEEncoder(self.dataset_B.feature_shapes["input"], self.n_latent).to(self.device)

        self.G_A = Generator(self.n_latent, self.dataset_A.feature_shapes["input"]).to(self.device)
        self.G_B = Generator(self.n_latent, self.dataset_B.feature_shapes["input"]).to(self.device)

        self.params_G = (list(self.E_A_fast.parameters()) 
                         + list(self.E_B_fast.parameters()) 
                         + list(self.G_A.parameters()) 
                         + list(self.G_B.parameters())
                         )
        
        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=1e-3)

        self.Dis_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.optimizer_Dis_m = optim.AdamW(self.Dis_Z.parameters(), lr=1e-3, weight_decay=1e-3)

        self.Dis_A = (
            MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device)
            if self.dataset_A.source_categories > 1
            else None
        )
        self.Dis_B = (
            MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device)
            if self.dataset_B.source_categories > 1
            else None
        )

        params = []
        if self.Dis_A:
            params += list(self.Dis_A.parameters())
        if self.Dis_B:
            params += list(self.Dis_B.parameters())

        self.optimizer_Dis_b = optim.AdamW(params, lr=1e-3, weight_decay=1e-3) if params else None


    def _init_low_encoders(self):
        self.E_A_slow = VAEEncoder(
            self.dataset_A.feature_shapes["input"], self.n_latent
        ).to(self.device)
        self.E_B_slow = VAEEncoder(
            self.dataset_B.feature_shapes["input"], self.n_latent
        ).to(self.device)

        self.E_A_slow.load_state_dict(self.E_A_fast.state_dict())
        self.E_B_slow.load_state_dict(self.E_B_fast.state_dict())

        for p in self.E_A_slow.parameters():
            p.requires_grad = False
        for p in self.E_B_slow.parameters():
            p.requires_grad = False


    def update_slow_encoder(self):
        self.E_A_slow.load_state_dict(self.E_A_fast.state_dict())
        self.E_B_slow.load_state_dict(self.E_B_fast.state_dict())


    def _set_train_mode(self):
        for m in [
            self.E_A_fast,
            self.E_B_fast,
            self.G_A,
            self.G_B,
            self.Dis_Z,
            self.Dis_A,
            self.Dis_B,
        ]:
            if m:
                m.train()


    def _set_eval_mode(self):
        for m in [
            self.E_A_fast,
            self.E_B_fast,
            self.G_A,
            self.G_B,
            self.Dis_Z,
            self.Dis_A,
            self.Dis_B,
        ]:
            if m:
                m.eval()


    def get_latent_representation(self):
        self._set_eval_mode()
        begin_time = time.time()

        x_A = torch.stack([self.dataset_A[i]["input"] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]["input"] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A_fast(x_A)
            _, mu_B, _ = self.E_B_fast(x_B)

        self.latent = np.concatenate([mu_A.cpu().numpy(), mu_B.cpu().numpy()], axis=0)

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f}s")
        print(f"Latent shape: {self.latent.shape}")


    def update_shared_mask(
        self,
        resolution=1.0,
        min_shared_frac=0.05,
        min_similarity=0.9,
    ):
        print("Updating shared mask...")

        self._set_eval_mode()

        x_A = torch.stack([self.dataset_A[i]["input"] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]["input"] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A_fast(x_A)
            _, mu_B, _ = self.E_B_fast(x_B)

        self.is_shared_A, self.is_shared_B = leiden_shared_mask(
            z_A=mu_A.cpu().numpy(),
            z_B=mu_B.cpu().numpy(),
            resolution=resolution,
            min_shared_frac=min_shared_frac,
            min_similarity=min_similarity,
            device=self.device,
        )

        print(f"Shared cells: A={self.is_shared_A.sum().item()} "f"B={self.is_shared_B.sum().item()}")


    def get_imputation(self):
        self._set_eval_mode()
        begin_time = time.time()

        print(f"Imputation started at: {time.asctime()}")

        x_A = torch.stack([self.dataset_A[i]["input"] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]["input"] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A_fast(x_A)
            _, mu_B, _ = self.E_B_fast(x_B)
            x_AtoB = self.G_B(mu_A)
            x_BtoA = self.G_A(mu_B)

        self.imputed_AtoB = x_AtoB.cpu().numpy()
        self.imputed_BtoA = x_BtoA.cpu().numpy()

        end_time = time.time()
        print(f"Completed at: {time.asctime()}")
        print(f"Total time: {end_time - begin_time:.2f}s")


    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)

        state = {
            "E_A": self.E_A_fast.state_dict(),
            "E_B": self.E_B_fast.state_dict(),
            "G_A": self.G_A.state_dict(),
            "G_B": self.G_B.state_dict(),
        }

        torch.save(state, os.path.join(model_path, "ckpt.pth"))
        print(f"Model saved to {model_path}/ckpt.pth")