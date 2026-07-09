"""Training utilities and custom losses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.structures import SequenceSample
from src.utils import get_cached_edge_index, iter_sequence_minibatches, stack_sequence_batch


class CustomLoss:
    """Interval-aware loss combining midpoint and range accuracy."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.criterion_center = nn.HuberLoss(delta=1.0)
        self.criterion_range = nn.L1Loss()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_center = self.criterion_center(pred.mean(dim=1), target.mean(dim=1))
        predicted_range = pred[:, 1] - pred[:, 0]
        target_range = target[:, 1] - target[:, 0]
        loss_range = self.criterion_range(predicted_range, target_range) + torch.relu(-predicted_range).mean()
        return loss_center + self.alpha * loss_range


class CorrelationLoss:
    """Loss that balances point accuracy with cross-stock trend ranking."""

    def __init__(self, weight_mse: float = 0.3, weight_corr: float = 0.6, weight_penalty: float = 0.1):
        self.criterion_mse = nn.MSELoss()
        self.weight_mse = weight_mse
        self.weight_corr = weight_corr
        self.weight_penalty = weight_penalty

    @staticmethod
    def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - torch.mean(pred)
        y = target - torch.mean(target)
        numerator = torch.sum(x * y)
        denominator = torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + 1e-8
        return 1 - (numerator / denominator)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
        pred_min = pred[:, 0]
        true_min = target[:, 0]

        loss_mse = self.criterion_mse(pred, target)
        loss_corr_min = self.correlation_loss(pred_min, true_min)
        risk_penalty = torch.relu(pred_min - true_min).mean()
        loss = (
            self.weight_mse * loss_mse
            + self.weight_corr * loss_corr_min
            + self.weight_penalty * risk_penalty
        )

        if return_components:
            return {
                "total_loss": float(loss.item()),
                "mse_loss": float(loss_mse.item()),
                "correlation_loss": float(loss_corr_min.item()),
                "risk_penalty": float(risk_penalty.item()),
                "weight_mse": self.weight_mse,
                "weight_corr": self.weight_corr,
                "weight_penalty": self.weight_penalty,
            }
        return loss


class Trainer:
    """Train a graph-temporal forecasting model."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn: str = "huber",
        scheduler=None,
        logger=None,
        loss_weights: dict | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device or torch.device("cpu")
        self.loss_fn = loss_fn
        self.loss_history: list[float] = []

        if loss_fn == "huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_fn == "custom":
            alpha = 0.5 if loss_weights is None else loss_weights.get("alpha", 0.5)
            self.criterion = CustomLoss(alpha=alpha)
        elif loss_fn == "correlation":
            kwargs = loss_weights or {}
            filtered_kwargs = {
                key: kwargs[key]
                for key in ("weight_mse", "weight_corr", "weight_penalty")
                if key in kwargs
            }
            self.criterion = CorrelationLoss(**filtered_kwargs)
        else:
            self.criterion = nn.MSELoss()

        self.use_amp = self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None
        self._edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor] = {}
        self._batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor] = {}

    @staticmethod
    def _flatten_for_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.dim() == 3:
            return prediction.reshape(-1, prediction.size(-1)), target.reshape(-1, target.size(-1))
        return prediction, target

    def train_epoch(
        self,
        train_sequences: list[SequenceSample],
        log_components: bool = False,
        batch_size: int | None = None,
    ):
        """Train the model for one epoch."""

        if not train_sequences:
            raise ValueError("Training sequences are empty.")

        self.model.train()
        epoch_loss = 0.0
        sample_count = 0
        aggregated_components = None
        epoch_source = train_sequences
        if (
            isinstance(train_sequences, list)
            and train_sequences
            and isinstance(train_sequences[0], SequenceSample)
            and batch_size is not None
            and batch_size > 1
        ):
            epoch_source = iter_sequence_minibatches(train_sequences, batch_size)

        for batch_index, batch in enumerate(epoch_source):
            batch_samples = [batch] if isinstance(batch, SequenceSample) else list(batch)
            if not batch_samples:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            batched_inputs = stack_sequence_batch(
                batch_samples,
                device=self.device,
                edge_index_cache=self._edge_index_cache,
                batched_edge_index_cache=self._batched_edge_index_cache,
            )

            if batched_inputs is not None:
                x_batch, y_batch, edge_index = batched_inputs
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    prediction = self.model(x_batch, edge_index)
                    prediction_for_loss, target_for_loss = self._flatten_for_loss(prediction, y_batch)
                    if self.loss_fn == "correlation":
                        loss = self.criterion(prediction_for_loss, target_for_loss, return_components=False)
                    else:
                        loss = self.criterion(prediction_for_loss, target_for_loss)

                if log_components and self.loss_fn == "correlation":
                    with torch.no_grad():
                        component_dict = self.criterion(prediction_for_loss, target_for_loss, return_components=True)
                    weighted_components = {
                        key: (value * len(batch_samples) if isinstance(value, (int, float)) else value)
                        for key, value in component_dict.items()
                    }
                    if aggregated_components is None:
                        aggregated_components = weighted_components
                    else:
                        for key, value in weighted_components.items():
                            if isinstance(value, (int, float)):
                                aggregated_components[key] += value

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss * len(batch_samples)
                sample_count += len(batch_samples)
                continue

            for sample in batch_samples:
                x_seq = sample.x_seq.to(self.device)
                y_target = sample.y_target.to(self.device)
                if sample.static_edge_index is not None and sample.dynamic_edge_index is not None:
                    edge_index = (
                        get_cached_edge_index(sample.static_edge_index, self.device, self._edge_index_cache),
                        get_cached_edge_index(sample.dynamic_edge_index, self.device, self._edge_index_cache),
                    )
                else:
                    edge_index = get_cached_edge_index(sample.edge_index, self.device, self._edge_index_cache)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    prediction = self.model(x_seq, edge_index)
                    if self.loss_fn == "correlation":
                        loss = self.criterion(prediction, y_target, return_components=False)
                    else:
                        loss = self.criterion(prediction, y_target)

                sample_loss = float(loss.item())
                loss = loss / len(batch_samples)

                if log_components and self.loss_fn == "correlation":
                    with torch.no_grad():
                        component_dict = self.criterion(prediction, y_target, return_components=True)
                    if aggregated_components is None:
                        aggregated_components = component_dict.copy()
                    else:
                        for key, value in component_dict.items():
                            if isinstance(value, (int, float)):
                                aggregated_components[key] += value

                self.scaler.scale(loss).backward()
                batch_loss += sample_loss
                sample_count += 1

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += batch_loss

        average_loss = epoch_loss / max(1, sample_count)
        self.loss_history.append(average_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        if log_components and aggregated_components is not None:
            for key, value in list(aggregated_components.items()):
                if isinstance(value, (int, float)):
                    aggregated_components[key] = value / max(1, sample_count)
            aggregated_components["total_loss"] = average_loss
            return aggregated_components

        return average_loss

    def train(
        self,
        train_sequences: list[SequenceSample],
        num_epochs: int = 150,
        print_every: int = 50,
        early_stopping_patience: int = 30,
        warmup_epochs: int = 5,
        min_delta: float = 1e-4,
        batch_size: int | None = None,
    ) -> None:
        """Run the full optimisation loop."""

        base_lr = self.optimizer.param_groups[0]["lr"]

        for epoch in range(num_epochs):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            log_components = self.logger is not None and (epoch % 10 == 0 or epoch == num_epochs - 1)
            result = self.train_epoch(train_sequences, log_components=log_components, batch_size=batch_size)

            if isinstance(result, dict):
                average_loss = result["total_loss"]
                if self.logger is not None:
                    self.logger.log_loss_components(epoch + 1, result)
            else:
                average_loss = result
                if self.logger is not None:
                    self.logger.log_epoch(epoch + 1, {"loss": average_loss})

            improvement = self.best_loss - average_loss
            if improvement > min_delta:
                self.best_loss = average_loss
                self.patience_counter = 0
                self.best_model_state = {
                    key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1

            if (epoch + 1) % max(1, print_every) == 0:
                message = f"Epoch {epoch + 1:03d} | Loss: {average_loss:.6f}"
                if isinstance(result, dict):
                    message += (
                        f" | MSE: {result.get('mse_loss', 0.0):.4f}"
                        f" | Corr: {result.get('correlation_loss', 0.0):.4f}"
                    )
                if self.patience_counter > 0:
                    message += f" | Patience: {self.patience_counter}/{early_stopping_patience}"
                print(message)

            if self.patience_counter >= early_stopping_patience:
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                break

    def plot_loss(self, save_path: str | Path | None = None, show: bool = False) -> None:
        """Save the training curve."""

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(self.loss_history, label="Training Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("GAT-LSTM Training Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(figure)
