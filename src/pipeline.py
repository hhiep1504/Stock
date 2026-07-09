"""End-to-end experiment preparation and training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import ExperimentConfig
from src.data import DataLoader, FeatureEngineer
from src.evaluation import Evaluator, Visualizer
from src.feature_sets import FEATURE_SET_DEFINITIONS
from src.graph import GraphConstructor
from src.models import build_model
from src.structures import ExperimentRunResult, PreparedDataset, SequenceSample
from src.tracking import ExperimentLogger
from src.training import Trainer
from src.utils import resolve_device, set_seed


def _merge_edge_indices(static_edge_index: torch.Tensor | None, dynamic_edge_index: torch.Tensor) -> torch.Tensor:
    if static_edge_index is None or static_edge_index.numel() == 0:
        return dynamic_edge_index
    if dynamic_edge_index.numel() == 0:
        return static_edge_index
    return torch.unique(torch.cat([static_edge_index, dynamic_edge_index], dim=1), dim=1)


def resolve_sequence_split_point(num_sequences: int, split_idx: int) -> int:
    """Resolve a signed split index against the total sequence count."""

    split_point = split_idx if split_idx >= 0 else num_sequences + split_idx
    if split_point <= 0 or split_point >= num_sequences:
        raise ValueError(f"Invalid split_idx={split_idx} for {num_sequences} sequences.")
    return split_point


def materialise_sequences_from_templates(
    x_full: torch.Tensor,
    y_full: torch.Tensor,
    sequence_templates: list[SequenceSample],
    window_size: int,
    sequence_indices: list[int] | None = None,
) -> list[SequenceSample]:
    """Attach scaled temporal slices to graph templates without rebuilding edges."""

    indices = sequence_indices if sequence_indices is not None else list(range(len(sequence_templates)))
    sequences: list[SequenceSample] = []

    for start_index in indices:
        template = sequence_templates[start_index]
        sequences.append(
            SequenceSample(
                x_seq=x_full[start_index : start_index + window_size],
                y_target=y_full[start_index + window_size],
                edge_index=template.edge_index,
                label=template.label,
                static_edge_index=template.static_edge_index,
                dynamic_edge_index=template.dynamic_edge_index,
            )
        )

    return sequences


def scale_sequences_for_train_count(
    feature_engineer: FeatureEngineer,
    x_full_raw: torch.Tensor,
    y_full: torch.Tensor,
    sequence_templates: list[SequenceSample],
    window_size: int,
    train_sequence_count: int,
    sequence_indices: list[int] | None = None,
) -> tuple[torch.Tensor, list[SequenceSample]]:
    """Fit a scaler on the train prefix, transform chronologically, and materialise sequences."""

    scaler = feature_engineer.fit_scaler_on_train_sequences(
        x_full_raw,
        train_sequence_count=train_sequence_count,
        window_size=window_size,
    )
    x_full_scaled = feature_engineer.transform_with_scaler(x_full_raw, scaler)
    sequences = materialise_sequences_from_templates(
        x_full=x_full_scaled,
        y_full=y_full,
        sequence_templates=sequence_templates,
        window_size=window_size,
        sequence_indices=sequence_indices,
    )
    return x_full_scaled, sequences


def create_sequences(
    x_full: torch.Tensor,
    y_full: torch.Tensor,
    data_loader: DataLoader,
    graph_constructor: GraphConstructor,
    valid_indices_map: list[Any],
    graph_mode: str = "hybrid",
    window_size: int = 2,
    top_k: int = 5,
    aggregation_mode: str = "quarterly",
    static_edge_index: torch.Tensor | None = None,
    corr_threshold: float = 0.7,
    similarity_metric: str = "cosine",
    use_arm: bool = False,
) -> list[SequenceSample]:
    """Create temporal graph samples from the full tensor sequence."""

    if graph_mode not in {"static", "dynamic", "hybrid", "dual_graph"}:
        raise ValueError("graph_mode must be 'static', 'dynamic', 'hybrid', or 'dual_graph'.")

    sequences: list[SequenceSample] = []
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    for start_index in range(len(x_full) - window_size):
        seq_x = x_full[start_index : start_index + window_size]
        seq_y = y_full[start_index + window_size]

        if aggregation_mode == "weekly":
            start_week = valid_indices_map[start_index]
            end_week = valid_indices_map[start_index + window_size - 1]
            subset_prices = data_loader.get_week_range(start_week, end_week)
        else:
            start_year, start_quarter = valid_indices_map[start_index]
            end_year, end_quarter = valid_indices_map[start_index + window_size - 1]
            subset_prices = data_loader.get_date_range(start_year, start_quarter, end_year, end_quarter)

        if graph_mode == "static":
            edge_index = static_edge_index if static_edge_index is not None else empty_edge_index
            static_component = None
            dynamic_component = None
        else:
            dynamic_edge_index = graph_constructor.create_dynamic_graph(
                subset_prices,
                top_k=top_k,
                use_arm=use_arm,
                corr_threshold=corr_threshold,
                similarity_metric=similarity_metric,
            )
            if graph_mode == "dynamic":
                edge_index = dynamic_edge_index
                static_component = None
                dynamic_component = None
            elif graph_mode == "dual_graph":
                static_component = static_edge_index if static_edge_index is not None else empty_edge_index
                dynamic_component = dynamic_edge_index
                edge_index = _merge_edge_indices(static_component, dynamic_component)
            else:
                edge_index = _merge_edge_indices(static_edge_index, dynamic_edge_index)
                static_component = None
                dynamic_component = None
        sequences.append(
            SequenceSample(
                x_seq=seq_x,
                y_target=seq_y,
                edge_index=edge_index,
                label=valid_indices_map[start_index + window_size],
                static_edge_index=static_component,
                dynamic_edge_index=dynamic_component,
            )
        )

    return sequences


def split_sequences(sequences: list[SequenceSample], split_idx: int) -> tuple[list[SequenceSample], list[SequenceSample]]:
    """Split the sequence list into train and test partitions."""

    if not sequences:
        raise ValueError("No sequences were created. Check the dataset and window size.")

    split_point = resolve_sequence_split_point(len(sequences), split_idx)
    return sequences[:split_point], sequences[split_point:]


def prepare_dataset(config: ExperimentConfig) -> PreparedDataset:
    """Run data loading, feature engineering, and graph preparation."""

    config.resolve_paths()
    data_loader = DataLoader(config.data.daily_file, config.data.target_file)
    daily_frame = data_loader.load_daily_data()
    stock_codes = data_loader.get_stock_codes()

    feature_engineer = FeatureEngineer(daily_frame, stock_codes, aggregation_mode=config.data.aggregation_mode)
    feature_names = list(FEATURE_SET_DEFINITIONS[config.data.feature_set])
    feature_frames = feature_engineer.compute_feature_frames(config.data.feature_set)
    target_min, target_max = feature_engineer.compute_targets()
    x_full_raw, y_full, valid_indices_map = feature_engineer.create_tensors_from_features(
        feature_frames=feature_frames,
        target_min=target_min,
        target_max=target_max,
        feature_names=feature_names,
    )
    config.model.in_features = int(x_full_raw.size(-1))

    graph_constructor = GraphConstructor(stock_codes)
    static_edge_index = None
    if config.graph.uses_static_component():
        if config.graph.sector_map_file is not None:
            graph_constructor.sector_map = GraphConstructor.load_sector_map(config.graph.sector_map_file)
            graph_constructor.use_bridge_edges = False
        graph_constructor.sector_map = graph_constructor.get_sector_mapping()
        static_edge_index = graph_constructor.create_static_graph()

    sequence_templates = create_sequences(
        x_full=x_full_raw,
        y_full=y_full,
        data_loader=data_loader,
        graph_constructor=graph_constructor,
        valid_indices_map=valid_indices_map,
        graph_mode=config.graph.graph_mode,
        window_size=config.data.window_size,
        top_k=config.graph.top_k,
        aggregation_mode=config.data.aggregation_mode,
        static_edge_index=static_edge_index,
        corr_threshold=config.graph.corr_threshold,
        similarity_metric=config.graph.similarity_metric,
        use_arm=config.graph.use_arm,
    )
    split_point = resolve_sequence_split_point(len(sequence_templates), config.data.split_idx)
    x_full, sequences = scale_sequences_for_train_count(
        feature_engineer=feature_engineer,
        x_full_raw=x_full_raw,
        y_full=y_full,
        sequence_templates=sequence_templates,
        window_size=config.data.window_size,
        train_sequence_count=split_point,
    )

    return PreparedDataset(
        daily_frame=daily_frame,
        stock_codes=stock_codes,
        x_full=x_full,
        x_full_raw=x_full_raw,
        y_full=y_full,
        feature_names=feature_names,
        valid_indices_map=valid_indices_map,
        static_edge_index=static_edge_index,
        sequence_templates=sequence_templates,
        sequences=sequences,
    )


def run_training_experiment(config: ExperimentConfig, dry_run: bool = False) -> ExperimentRunResult:
    """Run the full GAT-LSTM experiment pipeline."""

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    device = resolve_device(config.runtime.device)
    prepared = prepare_dataset(config)
    train_sequences, test_sequences = split_sequences(prepared.sequences, config.data.split_idx)

    if dry_run:
        return ExperimentRunResult(
            experiment_dir=None,
            metrics={
                "status": "dry_run",
                "device": str(device),
                "num_sequences": len(prepared.sequences),
                "num_stocks": len(prepared.stock_codes),
            },
            train_sequences=len(train_sequences),
            test_sequences=len(test_sequences),
            stock_codes=prepared.stock_codes,
        )

    experiment_name = config.experiment_name or config.build_experiment_name()
    logger = ExperimentLogger(log_dir=config.paths.logs_dir, experiment_name=experiment_name)
    experiment_dir = None
    results: dict[str, Any] = {}
    range_metrics: dict[str, Any] = {}
    try:
        logger.log_config(config.to_dict())

        model = build_model(config.model, num_nodes=len(prepared.stock_codes)).to(device)
        logger.log_model_info(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = None
        if config.training.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=config.loss.name,
            scheduler=scheduler,
            logger=logger,
            loss_weights=config.loss.as_kwargs(),
            device=device,
        )
        trainer.train(
            train_sequences,
            num_epochs=config.training.epochs,
            print_every=config.training.print_every,
            early_stopping_patience=config.training.early_stopping_patience,
            warmup_epochs=config.training.warmup_epochs,
            min_delta=config.training.min_delta,
            batch_size=config.training.batch_size,
        )

        experiment_dir = logger.get_experiment_dir()
        if config.runtime.save_plots:
            trainer.plot_loss(
                save_path=experiment_dir / "training_loss.png",
                show=config.runtime.show_plots,
            )

        evaluator = Evaluator(model, device=device)
        results = evaluator.evaluate(test_sequences)
        evaluator.print_metrics(results)
        logger.log_final_results(results)

        if config.runtime.save_plots:
            visualizer = Visualizer(prepared.stock_codes, aggregation_mode=config.data.aggregation_mode)
            visualizer.plot_predictions(
                results["predictions"],
                results["targets"],
                prepared.valid_indices_map,
                len(train_sequences),
                save_dir=experiment_dir,
                show=config.runtime.show_plots,
            )
            range_metrics = visualizer.analyze_range_compression(
                results["predictions"],
                results["targets"],
                save_path=experiment_dir / "range_analysis.png",
                show=config.runtime.show_plots,
            )
        else:
            range_metrics = {}

        logger.create_summary(
            additional_notes=(
                f"Train sequences: {len(train_sequences)} | "
                f"Test sequences: {len(test_sequences)} | "
                f"Range ratio: {range_metrics.get('range_ratio', 'N/A')}"
            )
        )

        if config.runtime.save_checkpoint:
            logger.save_checkpoint(model, optimizer, len(trainer.loss_history))
    finally:
        logger.close()

    return ExperimentRunResult(
        experiment_dir=experiment_dir,
        metrics={key: value for key, value in results.items() if key not in {"predictions", "targets"}},
        train_sequences=len(train_sequences),
        test_sequences=len(test_sequences),
        stock_codes=prepared.stock_codes,
    )
