
import torch
import datetime
import time
import sys,math
from torch.nn import functional as F
from utils.online_sampler import OnlineTestSampler
from torch.utils.data import DataLoader
from methods._trainer import _Trainer
from torchvision.utils import save_image


class AACL(_Trainer):
    def __init__(self, **kwargs):
        super(AACL, self).__init__(**kwargs)
        self.dtype = torch.double
        self.out_features = 0
        self.Gamma = kwargs.get("gamma_main") 
        self.buffer_size =  kwargs.get("buffer_size")
        self.feature_size = self.buffer_size
        self.eta=kwargs.get("eta")
        self.beta=kwargs.get("beta")
        self.mode_type=kwargs.get("mode_type")

    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        #print(self.model)
        for images, labels, idx in self.train_dataloader:
            #save_image(images, f'/home/block/continual/INCL/input_image{idx[0]}.png',nrow=5)
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
            #print("Train:",y.unique().shape)
            self.fit(X, y)
            logits = self.forward(X)
            _, pred_label = torch.max(logits, 1)
            acc = (pred_label == y).sum().item() / y.size(0) #  correct_cnt
            _acc += acc
            _iter += 1
        
        return _acc / _iter

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the classifier incrementally by the input features X and label y"""
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

        if self.mode_type=="direct":
            '''AACL main logic for direct updates'''
            lambda_fisher = torch.diag(self.W.pow(2).mean(dim=1))
            P_prev = torch.linalg.inv(self.R)
            P_new = P_prev + self.eta * lambda_fisher + self.beta * (X_fe.T @ X_fe)
            identity_D = torch.eye(P_new.shape[0], device=X_fe.device, dtype=torch.double)
            self.R = torch.linalg.inv(P_new + 1e-6 * identity_D)
            prior_mean_term = P_prev @ self.W + self.eta * (lambda_fisher @ self.W)
            new_evidence_term = self.beta * (X_fe.T @ Y)
            self.W = self.R @ (prior_mean_term + new_evidence_term)

        elif self.mode_type=="woodbury":
            # peak_mem = torch.cuda.max_memory_allocated(0) / (1024**2)
            # print(f"Peak memory during this batch: {peak_mem:.2f} MB")
            ''' AACL main logic for improving AACL Efficiency via Woodbury Updates'''
            N, D = X_fe.shape
            lambda_fisher_diag = self.W.pow(2).mean(dim=1)  # [D] vector
            R_prev = self.R
            eta_lambda = self.eta * lambda_fisher_diag
            R_diag = torch.diag(R_prev)
            factor = eta_lambda / (1.0 + eta_lambda * R_diag + 1e-10)
            R_scaled = R_prev * factor.unsqueeze(1)
            R_temp = R_prev - R_scaled @ R_prev
            sqrt_beta = torch.sqrt(torch.tensor(self.beta, device=self.device, dtype=torch.double))
            U = sqrt_beta * X_fe.T  # [D, N]

            R_temp_U = R_temp @ U  # [D, N]
            identity_N = torch.eye(N, device=self.device, dtype=torch.double)
            M = identity_N + U.T @ R_temp_U
            try:
                L = torch.linalg.cholesky(M)
                M_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                M_inv = torch.linalg.inv(M + 1e-8 * identity_N)
            G = R_temp_U @ M_inv  
            U_T_R_temp = U.T @ R_temp  
            self.R = R_temp - G @ U_T_R_temp 
            self.R = 0.5 * (self.R + self.R.T)
            try:
                L_prev = torch.linalg.cholesky(R_prev)
                y = torch.linalg.solve_triangular(L_prev, self.W, upper=False)
                R_prev_inv_W = torch.linalg.solve_triangular(L_prev.T, y, upper=True)
            except RuntimeError:
                R_prev_inv_W = torch.linalg.solve(R_prev, self.W)

            lambda_fisher = torch.diag(lambda_fisher_diag)
            b = R_prev_inv_W + self.eta * (lambda_fisher @ self.W) + self.beta * (X_fe.T @ Y)
            self.W = self.R @ b  # [D, K]

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
                #print("evale1",y.unique().shape)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                logit = self.forward(x)
                #print("exposed:",y.unique().shape)
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
