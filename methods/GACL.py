
import torch
import datetime
import time
import sys
from torch.nn import functional as F
from utils.online_sampler import OnlineTestSampler
from torch.utils.data import DataLoader
from methods._trainer import _Trainer


class GACL(_Trainer):
    def __init__(self, **kwargs):
        super(GACL, self).__init__(**kwargs)
        self.dtype = torch.double
        self.out_features = 0
        self.Gamma = kwargs.get("gamma_main") 
        self.buffer_size =  kwargs.get("buffer_size")
        self.feature_size = self.buffer_size

    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        #print(self.model)
        for images, labels, idx in self.train_dataloader:
        
            self.samples_cnt += images.size(0)

            acc = self.online_step(images, labels, idx)
            self.report_training(self.samples_cnt, acc)

            if self.samples_cnt > self.num_eval:
                with torch.no_grad():
                    test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                    test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                    eval_dict = self.online_evaluate(test_dataloader)
                    self.eval_results["test_acc"].append(eval_dict['avg_acc'])
                    self.eval_results["avg_acc"].append(eval_dict['cls_acc'])
                    self.eval_results["data_cnt"].append(self.samples_cnt)
                    self.report_test(self.samples_cnt, eval_dict['avg_acc'])
                    self.num_eval += self.eval_period
            sys.stdout.flush()
        if len(eval_dict)!= 0:
            self.report_test(self.samples_cnt, eval_dict['avg_acc'])


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_fe = self.model.expansion(X).double().to(self.device)
        return X_fe @ self.W

    def online_step(self, X, y, idx):
        """Use the data form data_loader to train the classifier incrementally."""
        self.add_new_class(y) 
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        
        _acc, _iter = 0.0, 0
        for _ in range(int(self.online_iter)):
            X = X.to(self.device)
            y = y.to(self.device)
            self.fit(X, y)
            logits = self.forward(X)
            _, pred_label = torch.max(logits, 1)
            acc = (pred_label == y).sum().item() / y.size(0) #  correct_cnt
            _acc += acc
            _iter += 1
        
        return _acc / _iter

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the classifier incrementally by the input features X and label y (integers, not one-hot)"""
        X_fe = self.model.expansion(X).double().to(self.device)
        num_classes = max(self.out_features, torch.max(y).item() + 1)
        assert isinstance(num_classes, int)
        if num_classes > self.out_features:
            increment_size = num_classes - self.out_features
            tail = torch.zeros((self.W.shape[0], increment_size)).to(self.W)
            self.W = torch.concat((self.W, tail), dim=1)
            self.out_features = num_classes


        Y = F.one_hot(y, self.out_features).double().to(self.device)
        self.R = self.R.to(self.device)
        self.W = self.W.to(self.device)
        #print("X_fe:",X_fe.shape,"Y:",Y.shape,"self.R",self.R.shape,"self.W",self.W.shape)

        #K = torch.eye(X_fe.shape[0]).to(X_fe) + X_fe @ self.R @ X_fe.T
        #self.R -= self.R @ X_fe.T @ torch.inverse(K) @ X_fe @ self.R
        #self.W += self.R @ X_fe.T @ (Y - X_fe @ self.W)
        '''
        lambda_reg = 1e-3  # Adjust as needed (e.g., based on dropout equivalence)
        batch_size = X_fe.shape[0]
        identity = torch.eye(batch_size, device=X_fe.device, dtype=torch.double)
        K = identity + X_fe @ self.R @ X_fe.T + lambda_reg * identity
        K_inv = torch.linalg.inv(K)  # Safe due to λ
        self.R -= self.R @ X_fe.T @ K_inv @ X_fe @ self.R
        eigvals, eigvecs = torch.linalg.eigh(self.R)
        eigvals = torch.clamp(eigvals, min=1e-5)
        self.R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        residual = Y - X_fe @ self.W
        delta_W = self.R @ X_fe.T @ residual
        self.W += delta_W
        '''
        ''' 
        gamma = 0.01
        trace_XRXT = torch.trace(X_fe @ self.R @ X_fe.T)
        lambda_reg = gamma * trace_XRXT / X_fe.shape[1]
        identity = torch.eye(X_fe.shape[0], device=X_fe.device, dtype=torch.double)
        K = (1 + lambda_reg) * identity + X_fe @ self.R @ X_fe.T
        K_inv = torch.linalg.inv(K)
        self.R -= self.R @ X_fe.T @ K_inv @ X_fe @ self.R
        eigvals, eigvecs = torch.linalg.eigh(self.R)
        eigvals = torch.clamp(eigvals, min=1e-5)
        self.R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        residual = Y - (1 + lambda_reg) * (X_fe @ self.W)
        delta_W = self.R @ X_fe.T @ residual 
        self.W += delta_W
        '''
        '''
        gamma = 0.01
        trace_XRXT = torch.trace(X_fe @ self.R @ X_fe.T)
        lambda_reg = gamma * trace_XRXT / X_fe.shape[1]
        identity = torch.eye(X_fe.shape[0], device=X_fe.device, dtype=torch.double)
        K = (1 + lambda_reg) * identity + X_fe @ self.R @ X_fe.T
        K_inv = torch.linalg.inv(K)
        stability_prior = torch.diag(self.W.abs().mean(dim=1) ** 2)  # shape: [D x D]
        eta = 0.01  # small coefficient to preserve useful directions
        self.R -= self.R @ X_fe.T @ K_inv @ X_fe @ self.R
        self.R += eta * stability_prior
        eigvals, eigvecs = torch.linalg.eigh(self.R)
        eigvals = torch.clamp(eigvals, min=1e-5)
        self.R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        residual = Y - (1 + lambda_reg) * (X_fe @ self.W)
        delta_W = self.R @ X_fe.T @ residual
        R_diag = torch.diag(self.R)  # [D]
        mask = R_diag / (R_diag + 1e-5)  # [D]
        delta_W = mask.unsqueeze(1) * delta_W  # [D x C]
        self.W += delta_W
        '''
        '''
        gamma = 0.01  # Adaptive regularization coefficient
        eta = 0.01    # Stability prior strength
        XRXT = X_fe @ self.R @ X_fe.T  # Shape: [N x N]
        trace_XRXT = torch.trace(XRXT)
        lambda_reg = gamma * trace_XRXT / X_fe.shape[1]  # Scalar
        batch_size = X_fe.shape[0]
        identity = torch.eye(batch_size, device=X_fe.device, dtype=torch.double)
        K = (1 + lambda_reg) * identity + XRXT  # Shape: [N x N]
        K_inv = torch.linalg.inv(K)
        self.R -= self.R @ X_fe.T @ K_inv @ X_fe @ self.R  # GACL-style update
        if self.W.shape[1] > 1:
            prior_diag = self.W.pow(2).mean(dim=1)  # mean across classes
        else:
         prior_diag = self.W.squeeze().pow(2)

        stability_prior = torch.diag(prior_diag)
        self.R += eta * stability_prior
        eigvals, eigvecs = torch.linalg.eigh(self.R)
        eigvals = torch.clamp(eigvals, min=1e-5)
        self.R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        residual = Y - (1 + lambda_reg) * (X_fe @ self.W)  # Shape: [N x C]
        delta_W = self.R @ X_fe.T @ residual  # Shape: [D x C]
        R_diag = torch.diag(self.R)  # Shape: [D]
        mask = R_diag / (R_diag + 1e-5)  # [D] in range (0, 1)
        delta_W = mask.unsqueeze(1) * delta_W  # Scale per-dimension
        self.W += delta_W
        '''

        eta = 10.0
        beta = 1.0

        if self.W.numel() > 0: # Check if weights are initialized
            lambda_fisher = torch.diag(self.W.pow(2).mean(dim=1)) # Shape: [D x D]
        else:
            lambda_fisher = torch.zeros((X_fe.shape[1], X_fe.shape[1]), device=X_fe.device, dtype=torch.double)

        try:
            P_prev = torch.linalg.inv(self.R)
        except torch.linalg.LinAlgError:
            P_prev = torch.zeros((X_fe.shape[1], X_fe.shape[1]), device=X_fe.device, dtype=torch.double)

        P_new = P_prev + eta * lambda_fisher + beta * (X_fe.T @ X_fe)
        identity_D = torch.eye(P_new.shape[0], device=X_fe.device, dtype=torch.double)
        self.R = torch.linalg.inv(P_new + 1e-6 * identity_D)
        if self.W.numel() > 0:
            prior_mean_term = P_prev @ self.W + eta * (lambda_fisher @ self.W)
        else:
            prior_mean_term = torch.zeros_like(X_fe.T @ Y, dtype=torch.double)

        new_evidence_term = beta * (X_fe.T @ Y)
        self.W = self.R @ (prior_mean_term + new_evidence_term)

        #assert torch.isfinite(K).all().item(),      "Pay attention to the numerical stability."
        assert torch.isfinite(self.R).all().item(), "Pay attention to the numerical stability."
        assert torch.isfinite(self.W).all().item(), "Pay attention to the numerical stability."

    def online_evaluate(self, test_loader):
        total_correct = 0.0
        total_num_data = 0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                logit = self.forward(x)

                pred = torch.argmax(logit, dim=-1)
                
                _, pred_label = logit.topk(self.topk, 1, True, True)


                total_correct += torch.sum(pred_label == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()

        avg_acc = total_correct / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_acc": avg_acc, "cls_acc": cls_acc}
        
        return eval_dict

    
    def online_after_task(self, task_id):
        pass

    def online_before_task(self, task_id):
        pass

    def setup_model(self):
        super().setup_model()
        self.W = torch.zeros((self.feature_size, 0)).double().to(self.device)
        # Autocorrelation Memory Matrix
        self.R = (torch.eye(self.feature_size) / self.Gamma).double().to(self.device)


    def report_training(self, sample_num, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_acc {train_acc:.4f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_acc):
        print(
            f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | "
        )
