import os
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # First derivatives (velocity)
    df['pitch_velocity'] = df['pitch'].diff() / df['tick'].diff()
    df['yaw_velocity'] = df['yaw'].diff() / df['tick'].diff()

    # Second derivatives (acceleration)
    df['pitch_acceleration'] = df['pitch_velocity'].diff() / df['tick'].diff()
    df['yaw_acceleration'] = df['yaw_velocity'].diff() / df['tick'].diff()

    # Third derivatives (jerk)
    df['pitch_jerk'] = df['pitch_acceleration'].diff() / df['tick'].diff()
    df['yaw_jerk'] = df['yaw_acceleration'].diff() / df['tick'].diff()

    # Cumulative angles
    df['cumulative_pitch'] = df['pitch'].cumsum()
    df['cumulative_yaw'] = df['yaw'].cumsum()

    # Euclidean distance between frames (angular "movement")
    df['angle_magnitude'] = np.sqrt(df['pitch'].diff()**2 + df['yaw'].diff()**2)

    # Smoothing to detect rapid directional changes
    df['yaw_change_sign'] = np.sign(df['yaw_velocity'].diff())
    df['pitch_change_sign'] = np.sign(df['pitch_velocity'].diff())
    df['direction_flips'] = (df['yaw_change_sign'].diff().abs() > 0).astype(int)
    df['flip_rate'] = df['direction_flips'].rolling(window=10).sum()

    # Moving average & rolling std (temporal smoothness/stability)
    df['yaw_rolling_std'] = df['yaw'].rolling(window=10, min_periods=1).std()
    df['pitch_rolling_std'] = df['pitch'].rolling(window=10, min_periods=1).std()
    df['yaw_rolling_mean'] = df['yaw'].rolling(window=10, min_periods=1).mean()
    df['pitch_rolling_mean'] = df['pitch'].rolling(window=10, min_periods=1).mean()

    # Peak detection features
    df['pitch_peaks'] = ((df['pitch_velocity'].diff().shift(-1) < 0) & 
                         (df['pitch_velocity'].diff() > 0)).astype(int)
    df['yaw_peaks'] = ((df['yaw_velocity'].diff().shift(-1) < 0) & 
                       (df['yaw_velocity'].diff() > 0)).astype(int)

    # Summary statistics (pitch/yaw)
    for col in ['pitch', 'yaw', 'pitch_velocity', 'yaw_velocity', 'angle_magnitude']:
        df[f'{col}_mean'] = df[col].mean()
        df[f'{col}_std'] = df[col].std()
        df[f'{col}_min'] = df[col].min()
        df[f'{col}_max'] = df[col].max()
        df[f'{col}_range'] = df[f'{col}_max'] - df[f'{col}_min']
        df[f'{col}_skew'] = df[col].skew()
        df[f'{col}_kurtosis'] = df[col].kurt()

    # Clean up NaNs
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
