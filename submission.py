import pandas as pd
import numpy as np
import os


def create_submission_file(X_test_index, final_preds, output_path):
    """Kaggle gönderim dosyasını oluşturur ve kaydeder."""
    submission_df = pd.DataFrame({
        'user_session': X_test_index,
        'session_value': final_preds
    })

    submission_df.to_csv(output_path, index=False)

    print("📊 Submission DataFrame oluşturuldu!")
    print(f"Shape: {submission_df.shape}")
    print(f"✅ Dosya başarıyla kaydedildi: {output_path}")

    return submission_df


def validate_submission(submission_df, required_cols=['user_session', 'session_value']):
    """Gönderim dosyasının formatını ve içeriğini doğrular."""
    print("\n🔍 DOSYA DOĞRULAMA:")
    print("-" * 40)

    # İstatistiksel kontrol
    print(f"📈 SESSION_VALUE İSTATİSTİKLERİ:")
    print(f"   Count: {submission_df['session_value'].count():,}")
    print(f"   Mean: {submission_df['session_value'].mean():.2f}")

    # Hata kontrolleri
    missing = submission_df.isnull().sum().sum()
    print(f"✅ Missing values: {missing}" if missing == 0 else f"⚠️ Missing values: {missing}")

    negative = (submission_df['session_value'] < 0).sum()
    print(f"✅ Negative values: {negative}" if negative == 0 else f"⚠️ Negative values: {negative}")

    duplicates = submission_df['user_session'].duplicated().sum()
    print(f"✅ Duplicate sessions: {duplicates}" if duplicates == 0 else f"⚠️ Duplicate sessions: {duplicates}")

    # Kaggle format kontrolü
    print(f"\n🎯 KAGGLE FORMAT KONTROL:")
    has_all_cols = all(col in submission_df.columns for col in required_cols)
    print("✅ Tüm gerekli sütunlar mevcut" if has_all_cols else "❌ Eksik sütunlar var!")

    if list(submission_df.columns) == required_cols:
        print("✅ Sütun sırası doğru")
    else:
        print("⚠️ Sütun sırası farklı")