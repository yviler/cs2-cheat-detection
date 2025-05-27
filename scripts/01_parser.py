import os
import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm

def parse_demo_folder(input_dir, cheater_ids, output_dir):
    base_dir = os.path.join(output_dir)
    os.makedirs(base_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Parsing demos"):
        if not filename.endswith('.dem'):
            continue
        demo_path = os.path.join(input_dir, filename)
        demo_base = os.path.splitext(filename)[0]

        print(f"Parsing {filename}...")
        try:
            parser = DemoParser(demo_path)
            events = parser.parse_event("player_death")
            df = parser.parse_ticks(["pitch", "yaw", "tick", "steamid"])

            for _, event in events.iterrows():
                start_tick = event["tick"] - 300
                end_tick = event["tick"]
                attacker = event.get("attacker_steamid")
                if not attacker:
                    continue

                attacker_int = int(attacker)
                label = 1 if attacker_int in cheater_ids else 0

                window = df[
                    df["tick"].between(start_tick, end_tick) &
                    (df["steamid"] == attacker_int)
                ]
                if window.empty:
                    continue

                window = window.drop_duplicates(subset="tick")
                full_index = list(range(start_tick, end_tick))
                window = (
                    window.set_index("tick")
                    .reindex(full_index)
                    .ffill()
                    .reset_index()
                    .rename(columns={"index": "tick"})
                )
                window["steamid"] = attacker_int
                window["label"] = label

                subfolder = "cheater" if label == 1 else "legit"
                user_dir = os.path.join(base_dir, subfolder, f"user_{attacker}")
                os.makedirs(user_dir, exist_ok=True)

                csv_name = f"{demo_base}_kill_{start_tick}_to_{end_tick}.csv"
                csv_path = os.path.join(user_dir, csv_name)
                window.to_csv(csv_path, index=False)
                print(f"  → Saved {csv_path}")

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

if __name__ == "__main__":
    cheater_ids = {
        76561199038314474,
        76561198119537465,
        76561199526241219,
        76561199773631959,
        76561199772547578,
        76561198900166541,
        76561199808250886,
        76561199787117228,
        76561198822643426,
        76561199847502271,
        76561199638564789,
        76561199809210656,
        76561198105396621,
        76561199484484936,
        76561199829535942,
        76561198987556574,
        76561199496724417,
        76561199849777985,
        76561199041458671,
    }

    output_dir = "data/interim/parsed_csv"

    for folder in ["mixed", "cheater", "legit"]:
        input_dir = os.path.join("data/raw/demos", folder)
        parse_demo_folder(input_dir=input_dir, cheater_ids=cheater_ids, output_dir=output_dir)
