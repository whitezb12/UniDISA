import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *
from typing import Optional, Dict, Literal
from itertools import chain, cycle
import os


class ImputationModel:
    def __init__(
        self,
        adata_A: "anndata.AnnData",
        adata_B: "anndata.AnnData",
        input_key: List[Optional[str]] = ['X_pca', 'X_lsi'],
        output_layer: List[Optional[str]] = ['counts', 'counts'],
        loss_type: List[Optional[str]] = ['l2', 'l2'],
        batch_size: int = 16,
        training_steps: int = 10000,
        seed: int = 1234,
        n_latent: int = 10,
        lambdaGuide: float = 1.0,
        lambdaRecon_x: float = 10.0,
        lambdaRecon_y: float = 10.0,
        lambdaLA: float = 10.0,
        model_path = "models"
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaGuide = lambdaGuide
        self.lambdaRecon_x = lambdaRecon_x
        self.lambdaRecon_y = lambdaRecon_y
        self.lambdaLA = lambdaLA

        self.dataset_A = AnnDataDataset(
            adata_A,
            input_key = input_key[0],
            output_layer = output_layer[0],
            mode="imputation"
        )
        self.dataset_B = AnnDataDataset(
            adata_B,
            input_key = input_key[1],
            output_layer = output_layer[1],
            mode="imputation"
        )
        
        self.dataloader_A = load_data(self.dataset_A, batch_size = self.batch_size, mode = "imputation")
        self.dataloader_B = load_data(self.dataset_B, batch_size = self.batch_size, mode = "imputation")

        self.loss_type = loss_type

        self.model_path = model_path
        

    def train(self) -> None:
        self._init_low_encoders()
        self._init_models_and_optimizers()
        self._set_train_mode()
        iterator_A = cycle(self.dataloader_A)
        iterator_B = cycle(self.dataloader_B)
        print("===== Start training imputation! =====")

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)
            y_A = batch_A['output'].float().to(self.device)
            y_B = batch_B['output'].float().to(self.device)

            _, link_A, _ = self.E_A_slow(x_A)
            _, link_B, _ = self.E_B_slow(x_B)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

            y_Arecon = self.D_A(x_Arecon)
            y_Brecon = self.D_B(x_Brecon)

            loss_dict = {}

            # guide loss
            loss_Guide = torch.mean((mu_A - link_A)**2) + torch.mean((mu_B - link_B)**2)

            # x reconstruction loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arecon - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE_x = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA  

            # y reconstruction loss            
            loss_AE_A = self.compute_Recony(y_Arecon, y_A, loss_type=self.loss_type[0], dispersion=self.dispersion_A)
            loss_AE_B = self.compute_Recony(y_Brecon, y_B, loss_type=self.loss_type[1], dispersion=self.dispersion_B)
            loss_AE_y = loss_AE_A + loss_AE_B

            total_loss = (
                self.lambdaGuide * loss_Guide
                + self.lambdaLA * loss_LA
                + self.lambdaRecon_y * loss_AE_y
                + self.lambdaRecon_x * loss_AE_x
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 500 == 0:
                print(
                    f"[Step {step}] "
                    f"Guide: {loss_Guide:.4f} | "
                    f"AE_x: {loss_AE_x:.4f} | "
                    f"LA: {loss_LA:.4f} | "
                    f"AE_y: {loss_AE_y:.4f}  "
                )
        

    def get_latent_representation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A_fast(x_A)
            _, mu_B, _ = self.E_B_fast(x_B)

        self.latent = np.concatenate((mu_A.cpu().numpy(), mu_B.cpu().numpy()), axis=0)
        end_time = time.time()

        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")


    def get_imputation(self, library_size: float = 1e4) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A_fast(x_A)
            _, mu_B, _ = self.E_B_fast(x_B)

        x_AtoB = self.G_B(mu_A)
        x_BtoA = self.G_A(mu_B)

        y_AtoB_logits = self.D_B(x_AtoB)
        y_BtoA_logits = self.D_A(x_BtoA)

        if self.loss_type[0] == 'nb':
            y_BtoA_scale = F.softmax(y_BtoA_logits, dim=-1)
            y_BtoA = y_BtoA_scale * library_size
        elif self.loss_type[0] == 'bce':
            y_BtoA = torch.sigmoid(y_BtoA_logits)
        else:
            y_BtoA = y_BtoA_logits
        
        if self.loss_type[1] == 'nb':
            y_AtoB_scale = F.softmax(y_AtoB_logits, dim=-1)
            y_AtoB = y_AtoB_scale * library_size
        elif self.loss_type[1] == 'bce':
            y_AtoB = torch.sigmoid(y_AtoB_logits)
        else:
            y_AtoB = y_AtoB_logits  

        self.imputed_AtoB = y_AtoB.detach().cpu().numpy()
        self.imputed_BtoA = y_BtoA.detach().cpu().numpy()

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")


    def _init_models_and_optimizers(self) -> None:
        self.E_A_fast = VAEEncoder(
            n_input=self.dataset_A.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.E_B_fast = VAEEncoder(
            n_input=self.dataset_B.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.G_A = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_A.feature_shapes['input']
        ).to(self.device)

        self.G_B = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_B.feature_shapes['input']
        ).to(self.device)

        self.D_A = Decoder(
            n_input=self.dataset_A.feature_shapes['input'],
            n_output=self.dataset_A.feature_shapes['output']
        ).to(self.device)

        self.D_B = Decoder(
            n_input=self.dataset_B.feature_shapes['input'],
            n_output=self.dataset_B.feature_shapes['output']
        ).to(self.device)

        if self.loss_type[0] == 'nb':
            self.dispersion_A = nn.Parameter(torch.rand(self.dataset_A.feature_shapes['output'], device=self.device))
        else:
            self.dispersion_A = None

        if self.loss_type[1] == 'nb':
            self.dispersion_B = nn.Parameter(torch.rand(self.dataset_B.feature_shapes['output'], device=self.device))
        else:
            self.dispersion_B = None

        self.params_G = chain(
            self.E_A_fast.parameters(),
            self.E_B_fast.parameters(),
            self.G_A.parameters(),
            self.G_B.parameters(),
            self.D_A.parameters(),
            self.D_B.parameters()
        )

        if self.dispersion_A is not None:
            self.params_G.append(self.dispersion_A)
        if self.dispersion_B is not None:
            self.params_G.append(self.dispersion_B)
        
        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=0.0)
    
    def _init_low_encoders(self) -> None:
        self.E_A_slow = VAEEncoder(
            n_input=self.dataset_A.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.E_B_slow = VAEEncoder(
            n_input=self.dataset_B.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        ckpt_path = os.path.join(self.model_path, "ckpt.pth")
        if self.model_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.E_A_slow.load_state_dict(checkpoint['E_A'])
            self.E_B_slow.load_state_dict(checkpoint['E_B'])
            print(f"✅ Loaded checkpoint from {ckpt_path}")
        else:
            print("⚠️ No checkpoint found, training from scratch.")

        for p in self.E_A_slow.parameters():
            p.requires_grad = False
        for p in self.E_B_slow.parameters():
            p.requires_grad = False
            

    def _set_train_mode(self) -> None:
        for model in [self.E_A_fast, self.E_B_fast, self.G_A, self.G_B, self.D_A, self.D_B]:
            if model is not None:
                model.train()


    def _set_eval_mode(self) -> None:
        for model in [self.E_A_fast, self.E_B_fast, self.G_A, self.G_B, self.D_A, self.D_B]:
            if model is not None:
                model.eval()


    def compute_Recony(self, approx_features, counts, loss_type, dispersion=None, eps=1e-8):
        if loss_type == 'nb':
            library_size = counts.sum(dim=1, keepdim=True)
            px_scale = F.softmax(approx_features, dim=-1)
            px_rate = px_scale * library_size
            if dispersion is None:
                raise ValueError("dispersion must be provided for NB loss")
            px_r = F.softplus(dispersion) + eps
            reconstruction_loss = -log_nb_positive(counts, px_rate, px_r).mean() / counts.shape[1]
        elif loss_type == 'bce':
            reconstruct = torch.sigmoid(approx_features)
            reconstruction_loss = F.binary_cross_entropy(
                reconstruct,
                counts,
                reduction='none'
            ).sum(-1).mean() * 10 / counts.shape[1]
        elif loss_type == 'l2':
            reconstruction_loss = torch.mean((approx_features - counts) ** 2)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        return reconstruction_loss




