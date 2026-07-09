"""
Check volatility of fold 3 across different tuning runs
Improved version that loads sequences properly
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, str(Path.cwd()))

from src.data import DataLoader
from src.data import FeatureEngineer


class Fold3VolatilityAnalyzer:
    def __init__(
        self,
        dataset_file: str = "dataset/stock_market_19_24.csv",
        target_file: str = "dataset/min_max_return.csv",
        logs_dir: str = "logs",
        window_size: int = 8,
        aggregation_mode: str = "weekly"
    ):
        self.dataset_file = dataset_file
        self.target_file = target_file
        self.logs_dir = Path(logs_dir)
        self.window_size = window_size
        self.aggregation_mode = aggregation_mode
        self.df_daily = None
        self.stock_codes = None
        self.valid_indices_map = None
        self.n_sequences = None
        
    def load_and_prepare_data(self):
        """Load stock data and prepare sequences info"""
        print("⏳ Loading and preparing data...")
        
        # Load data
        data_loader = DataLoader(self.dataset_file, self.target_file)
        self.df_daily = data_loader.load_daily_data()
        self.stock_codes = data_loader.get_stock_codes()
        print(f"✅ Loaded {len(self.df_daily)} trading days, {len(self.stock_codes)} stocks")
        
        # Feature engineering to get valid_indices_map
        feature_engineer = FeatureEngineer(
            self.df_daily, 
            self.stock_codes, 
            aggregation_mode=self.aggregation_mode
        )
        
        feat_std, feat_mean, feat_return, feat_skew = feature_engineer.compute_features()
        target_min, target_max = feature_engineer.compute_targets()
        x_full_raw, y_full, self.valid_indices_map = feature_engineer.create_tensors(
            feat_std,
            feat_mean,
            feat_return,
            feat_skew,
            target_min,
            target_max,
        )
        
        # Calculate number of sequences
        self.n_sequences = len(x_full_raw) - self.window_size
        print(f"✅ Total sequences: {self.n_sequences}")
        print(f"✅ Aggregation mode: {self.aggregation_mode}")
        print(f"✅ Window size: {self.window_size}")
        
    def get_time_labels(self) -> List[str]:
        """Convert valid_indices_map to readable time labels"""
        labels = []
        
        if self.aggregation_mode == "weekly":
            for idx in self.valid_indices_map:
                if isinstance(idx, tuple):  # (year, week)
                    year, week = idx
                    labels.append(f"{year}_W{week:02d}")  # Dấu gạch dưới, không gạch ngang
                else:
                    labels.append(str(idx))
        else:  # quarterly
            for idx in self.valid_indices_map:
                if isinstance(idx, tuple):  # (year, quarter)
                    year, quarter = idx
                    labels.append(f"{year}_Q{quarter}")  # Dấu gạch dưới
                else:
                    labels.append(str(idx))
        
        return labels
    
    def calculate_period_volatility(self) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate volatility for each time period"""
        print(f"⏳ Calculating {self.aggregation_mode} volatility...")
        
        # Group daily data by aggregation period
        period_vol = {}
        
        if self.aggregation_mode == "weekly":
            self.df_daily['Year'] = self.df_daily['Date'].dt.isocalendar().year
            self.df_daily['Week'] = self.df_daily['Date'].dt.isocalendar().week
            self.df_daily['YearWeek'] = (
                self.df_daily['Year'].astype(str) + 
                '_W' + 
                self.df_daily['Week'].astype(str).str.zfill(2)
            )
            group_by = 'YearWeek'
        else:  # quarterly
            self.df_daily['Year'] = self.df_daily['Date'].dt.year
            self.df_daily['Quarter'] = self.df_daily['Date'].dt.quarter
            self.df_daily['YearQuarter'] = (
                self.df_daily['Year'].astype(str) + 
                '_Q' + 
                self.df_daily['Quarter'].astype(str)
            )
            group_by = 'YearQuarter'
        
        # Calculate volatility for each period
        for period_label, group in self.df_daily.groupby(group_by):
            vols = []
            
            for stock in self.stock_codes:
                if stock not in group.columns:
                    continue
                prices = pd.to_numeric(group[stock], errors='coerce').dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    if len(returns) > 0:
                        vol = returns.std() * 100  # percentage
                        vols.append(vol)
            
            if vols:
                period_vol[period_label] = {
                    'mean_vol': np.mean(vols),
                    'max_vol': np.max(vols),
                    'min_vol': np.min(vols),
                    'std_vol': np.std(vols),
                    'stocks_count': len(vols)
                }
        
        df_vol = pd.DataFrame.from_dict(period_vol, orient='index')
        time_labels = self.get_time_labels()
        
        print(f"✅ Calculated volatility for {len(df_vol)} periods")
        
        return df_vol, time_labels
    
    def analyze_fold3_volatility(self, df_vol: pd.DataFrame, time_labels: List[str], n_folds: int = 5):
        """Analyze volatility of fold 3"""
        print("\n" + "="*80)
        print(f"FOLD 3 VOLATILITY ANALYSIS ({self.aggregation_mode.upper()})")
        print("="*80)
        
        # Calculate fold 3 boundaries
        fold_size = max(1, self.n_sequences // n_folds)
        fold_idx = 2  # fold 3 = index 2
        
        start_idx = fold_idx * fold_size
        end_idx = self.n_sequences if fold_idx == n_folds - 1 else (fold_idx + 1) * fold_size
        
        print(f"\nFold 3 Info:")
        print(f"  Total sequences: {self.n_sequences}")
        print(f"  Fold size: {fold_size}")
        print(f"  Fold 3 range: sequences [{start_idx}, {end_idx})")
        print(f"  Fold 3 size: {end_idx - start_idx} sequences")
        
        # Map fold 3 to time periods
        if start_idx < len(time_labels) and end_idx <= len(time_labels):
            fold_3_labels = time_labels[start_idx:end_idx]
            fold_3_vol = df_vol.loc[fold_3_labels]
            
            print(f"\nFold 3 Time Period:")
            print(f"  From: {fold_3_labels[0]}")
            print(f"  To:   {fold_3_labels[-1]}")
            print(f"  Periods: {len(fold_3_labels)}")
            
            # Volatility statistics
            print(f"\nFold 3 Volatility Statistics:")
            print(f"  Mean volatility: {fold_3_vol['mean_vol'].mean():.4f}%")
            print(f"  Median volatility: {fold_3_vol['mean_vol'].median():.4f}%")
            print(f"  Max volatility: {fold_3_vol['mean_vol'].max():.4f}%")
            print(f"  Min volatility: {fold_3_vol['mean_vol'].min():.4f}%")
            print(f"  Std Dev: {fold_3_vol['mean_vol'].std():.4f}%")
            
            # High volatility periods
            high_vol_threshold = fold_3_vol['mean_vol'].mean() + fold_3_vol['mean_vol'].std()
            high_vol_periods = fold_3_vol[fold_3_vol['mean_vol'] > high_vol_threshold]
            
            if len(high_vol_periods) > 0:
                print(f"\n🔥 High Volatility Periods ({len(high_vol_periods)}):")
                print(f"   (threshold: {high_vol_threshold:.4f}%)")
                for period, row in high_vol_periods.iterrows():
                    print(f"   {period}: {row['mean_vol']:.4f}%")
            
            # Low volatility periods
            low_vol_threshold = fold_3_vol['mean_vol'].mean() - fold_3_vol['mean_vol'].std()
            low_vol_periods = fold_3_vol[fold_3_vol['mean_vol'] < low_vol_threshold]
            
            if len(low_vol_periods) > 0:
                print(f"\n❄️  Low Volatility Periods ({len(low_vol_periods)}):")
                print(f"   (threshold: {low_vol_threshold:.4f}%)")
                for period, row in low_vol_periods.iterrows():
                    print(f"   {period}: {row['mean_vol']:.4f}%")
            
            # Save detailed results
            output_file = Path("outputs/fold3_volatility_details.csv")
            output_file.parent.mkdir(exist_ok=True)
            fold_3_vol.to_csv(output_file)
            print(f"\n✅ Detailed results saved to {output_file}")
            
        else:
            print(f"⚠️  Fold 3 indices out of range!")


if __name__ == "__main__":
    analyzer = Fold3VolatilityAnalyzer(
        dataset_file="dataset/stock_market_19_24.csv",
        target_file="dataset/min_max_return.csv",
        logs_dir="logs",
        window_size=8,
        aggregation_mode="weekly"
    )
    
    # Load and analyze
    analyzer.load_and_prepare_data()
    df_vol, time_labels = analyzer.calculate_period_volatility()
    analyzer.analyze_fold3_volatility(df_vol, time_labels, n_folds=5)