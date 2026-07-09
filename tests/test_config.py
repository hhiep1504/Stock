from pathlib import Path
import unittest

from src.config import ExperimentConfig, default_experiment_config


class ConfigSmokeTest(unittest.TestCase):
    def test_default_project_root_is_repository_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config = default_experiment_config()

        self.assertEqual(config.paths.project_root, repo_root)

    def test_default_experiment_json_loads(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default_experiment.json"
        config = ExperimentConfig.from_json(config_path)

        self.assertTrue(config.data.daily_file.name.endswith(".csv"))
        self.assertEqual(config.data.feature_set, "baseline4")


if __name__ == "__main__":
    unittest.main()
