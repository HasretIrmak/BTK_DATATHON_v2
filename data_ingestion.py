import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(base_path):
    """Veri setlerini yükler ve temel ayarları yapar."""
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

    print("Veri dosyaları başarıyla yüklendi.")
    print(f"Train veri boyutu: {train_df.shape}")
    print(f"Test veri boyutu: {test_df.shape}")

    return train_df, test_df


if __name__ == '__main__':
    # Örnek kullanım
    BASE_PATH = '/kaggle/input/datathon-2025'
    train_data, test_data = load_data(BASE_PATH)