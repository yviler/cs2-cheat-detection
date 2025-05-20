
import os
import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm

def parse_demo_folder(input_dir, label, output_dir):
    subfolder = 'cheater' if label == 1 else 'legit'
    base_dir = os.path.join(output_dir, subfolder)
    os.makedirs(base_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc=f"Parsing {subfolder} demos"):
        if not filename.endswith('.dem'):
            continue
        demo_path = os.path.join(input_dir, filename)
        demo_base = os.path.splitext(filename)[0]

        already = [f for f in os.listdir(base_dir) if f.startswith(demo_base)]
        if already:
            print(f"Skipping {filename}, already parsed.")
            continue

        try:
            parser = DemoParser(demo_path)
            events = parser.parse_event("player_death")
            df = parser.parse_ticks(["pitch", "yaw", "tick", "steamid"])
            df['label'] = label

            for _, event in events.iterrows():
                start_tick = event["tick"] - 300
                end_tick   = event["tick"]
                attacker   = event.get("attacker_steamid")
                if not attacker:
                    continue

                attacker_int = int(attacker)
                window = df[
                    df["tick"].between(start_tick, end_tick) &
                    (df["steamid"] == attacker_int)
                ]
                if window.empty:
                    continue

                window = window.drop_duplicates(subset="tick")
                full_index = list(range(start_tick, end_tick))
                window = (window
                          .set_index("tick")
                          .reindex(full_index)
                          .fillna(method="ffill")
                          .reset_index()
                          .rename(columns={"index": "tick"}))
                window['steamid'] = attacker_int
                window['label']   = label

                user_dir = os.path.join(base_dir, f"user_{attacker}")
                os.makedirs(user_dir, exist_ok=True)
                csv_name = f"{demo_base}_kill_{start_tick}_to_{end_tick}.csv"
                csv_path = os.path.join(user_dir, csv_name)
                window.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

def run_parse_all():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cheater_dir = os.path.join(base_dir, "data", "raw", "demos", "cheater")
    legit_dir   = os.path.join(base_dir, "data", "raw", "demos", "legit")
    output_dir  = os.path.join(base_dir, "data", "interim", "parsed_csv")

    print("Parsing cheater demos...")
    parse_demo_folder(cheater_dir, label=1, output_dir=output_dir)

    print("Parsing legit demos...")
    parse_demo_folder(legit_dir, label=0, output_dir=output_dir)

if __name__ == "__main__":
    run_parse_all()