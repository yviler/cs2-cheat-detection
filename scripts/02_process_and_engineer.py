import os
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    
    # First derivatives
    df['pitch_velocity'] = df['pitch'].diff() / df['tick'].diff()
    df['yaw_velocity'] = df['yaw'].diff() / df['tick'].diff()
    
    # Second derivatives
    df['pitch_acceleration'] = df['pitch_velocity'].diff() / df['tick'].diff()
    df['yaw_acceleration'] = df['yaw_velocity'].diff() / df['tick'].diff()
    
    # Third derivatives (jerk)
    df['pitch_jerk'] = df['pitch_acceleration'].diff() / df['tick'].diff()
    df['yaw_jerk'] = df['yaw_acceleration'].diff() / df['tick'].diff()
    
    # Cumulative displacement
    df['cumulative_pitch'] = df['pitch'].cumsum()
    df['cumulative_yaw'] = df['yaw'].cumsum()
    
    # Statistical summary
    for col in ['pitch', 'yaw']:
        df[f'{col}_mean'] = df[col].mean()
        df[f'{col}_std'] = df[col].std()
        df[f'{col}_min'] = df[col].min()
        df[f'{col}_max'] = df[col].max()
        df[f'{col}_range'] = df[f'{col}_max'] - df[f'{col}_min']
    
    return df.dropna()

def process_all_segments(input_dir, output_dir, category_name):
    os.makedirs(output_dir, exist_ok=True)
    for user_folder in os.listdir(input_dir):
        user_path = os.path.join(input_dir, user_folder)
        if not os.path.isdir(user_path):
            continue
        for file_name in os.listdir(user_path):
            if not file_name.endswith('.csv'):
                continue

            file_path = os.path.join(user_path, file_name)
            try:
                df = pd.read_csv(file_path)

                # Drop irrelevant columns
                df = df.drop(columns=[col for col in ['name'] if col in df.columns])

                processed = engineer_features(df)
                save_path = os.path.join(output_dir, f"engineered_{file_name}")
                processed.to_csv(save_path, index=False)
                print(f"✅ Processed: {save_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

if __name__ == "__main__":
    base_input = "data/interim/parsed_csv"
    base_output = "data/processed/features"

    cheater_input = os.path.join(base_input, "cheater")
    legit_input = os.path.join(base_input, "legit")
    cheater_output = os.path.join(base_output, "cheater")
    legit_output = os.path.join(base_output, "legit")

    # Process both categories
    print("⚙️ Processing cheater segments...")
    process_all_segments(cheater_input, cheater_output, "cheater")

    print("⚙️ Processing legit segments...")
    process_all_segments(legit_input, legit_output, "legit")
