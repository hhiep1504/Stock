"""
Configuration File
Centralized configuration for the GAT-LSTM model
"""

import os


class Config:
    """Configuration settings for GAT-LSTM stock prediction"""
    
    # ==========================================
    # DATA PATHS
    # ==========================================
    # Update these paths to match your setup
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    
    # Data files
    DAILY_FILE = os.path.join(DATA_DIR, 'stock_market_19_24.csv')
    TARGET_FILE = None  # Optional: quarterly min/max returns file
    
    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    # Architecture
    IN_FEATURES = 4  # Number of input features per node
    GNN_HIDDEN = 128  # Hidden dimension for GAT layers
    LSTM_HIDDEN = 256  # Hidden dimension for LSTM layer
    NUM_HEADS = 1  # Number of attention heads
    DROPOUT = 0.6  # Dropout rate
    
    # ==========================================
    # TRAINING PARAMETERS
    # ==========================================
    WINDOW_SIZE = 2  # Number of quarters for lookback
    NUM_EPOCHS = 200  # Number of training epochs
    LEARNING_RATE = 0.001  # Learning rate
    WEIGHT_DECAY = 1e-5  # L2 regularization
    
    # Loss function: 'huber', 'custom', 'correlation'
    LOSS_FUNCTION = 'correlation'
    
    # Scheduler settings
    USE_SCHEDULER = True
    SCHEDULER_T_MAX = 150  # CosineAnnealingLR parameter
    
    # ==========================================
    # GRAPH CONSTRUCTION
    # ==========================================
    TOP_K = 4  # Number of neighbors for KNN graph
    USE_ARM = True  # Use Association Rule Mining
    USE_STATIC_GRAPH = False  # Use sector-based static graph

    # ----------------------------------------------------------
    # Similarity metric for dynamic graph construction.
    # Options:
    #   'pearson'  – Pearson correlation (mean-centred, classic)
    #   'cosine'   – Cosine similarity (direction-based, faster)
    # ----------------------------------------------------------
    SIMILARITY_METRIC = 'pearson'  # <-- tune this

    # ----------------------------------------------------------
    # Correlation / similarity threshold τ (hyperparameter).
    # Only edges with score ≥ τ are kept in the dynamic graph.
    # Suggested sweep: {0.7, 0.8, 0.9}
    # Higher τ → sparser, less noisy graph.
    # ----------------------------------------------------------
    CORR_THRESHOLD = 0.6  # <-- tune this: try 0.7 / 0.8 / 0.9
    
    # ==========================================
    # DATA SPLIT
    # ==========================================
    SPLIT_IDX = -4  # Last N quarters for testing
    
    # ==========================================
    # SECTOR MAPPING (Vietnamese Stocks)
    # ==========================================
    SECTOR_MAP = {
        # Housing sector
        "VHM": "Housing", "NVL": "Housing", "PDR": "Housing", 
        "NLG": "Housing", "KDH": "Housing", "DXG": "Housing",
        "VPI": "Housing", "NTL": "Housing", "DIG": "Housing",
        "TCH": "Housing", "CRE": "Housing", "CCL": "Housing",
        "HDC": "Housing", "ITC": "Housing",
        
        # Industrial sector
        "LHG": "Industrial", "SZL": "Industrial", "TIP": "Industrial",
        "TIX": "Industrial", "KBC": "Industrial",
        
        # Construction sector
        "TDC": "Construction", "HDG": "Construction", 
        "CDC": "Construction", "D2D": "Construction",
    }
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    SAVE_PLOTS = True  # Whether to save plots to file
    DPI = 300  # Plot resolution
    FIGURE_SIZE = (12, 6)  # Default figure size
    
    # ==========================================
    # LOGGING
    # ==========================================
    PRINT_EVERY = 50  # Print training progress every N epochs
    VERBOSE = True  # Enable verbose logging
    
    # ==========================================
    # RANDOM SEED
    # ==========================================
    RANDOM_SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"✅ Directories created/verified:")
        print(f"   - Data: {cls.DATA_DIR}")
        print(f"   - Output: {cls.OUTPUT_DIR}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*70)
        print("⚙️ CONFIGURATION")
        print("="*70)
        print(f"Model Architecture:")
        print(f"   - Input Features: {cls.IN_FEATURES}")
        print(f"   - GNN Hidden: {cls.GNN_HIDDEN}")
        print(f"   - LSTM Hidden: {cls.LSTM_HIDDEN}")
        print(f"   - Attention Heads: {cls.NUM_HEADS}")
        print(f"\nTraining:")
        print(f"   - Window Size: {cls.WINDOW_SIZE}")
        print(f"   - Epochs: {cls.NUM_EPOCHS}")
        print(f"   - Learning Rate: {cls.LEARNING_RATE}")
        print(f"   - Loss Function: {cls.LOSS_FUNCTION}")
        print(f"\nGraph Construction:")
        print(f"   - Top-K Neighbors: {cls.TOP_K}")
        print(f"   - Use ARM: {cls.USE_ARM}")
        print(f"   - Use Static Graph: {cls.USE_STATIC_GRAPH}")
        print("="*70)


# Alternative configurations for experimentation

class ConfigDeepModel(Config):
    """Configuration for deeper model with more layers"""
    GNN_HIDDEN = 64
    LSTM_HIDDEN = 128
    NUM_HEADS = 4
    DROPOUT = 0.3
    NUM_EPOCHS = 300


class ConfigFastTrain(Config):
    """Configuration for faster training"""
    GNN_HIDDEN = 32
    LSTM_HIDDEN = 64
    NUM_HEADS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005


class ConfigLargeWindow(Config):
    """Configuration with larger lookback window"""
    WINDOW_SIZE = 4
    NUM_EPOCHS = 200
    TOP_K = 5


from src.config import ExperimentConfig as ResearchExperimentConfig
from src.config import default_experiment_config


def get_research_config() -> ResearchExperimentConfig:
    """Return the typed configuration used by the refactored pipeline."""

    return default_experiment_config()
