import os
import pandas as pd
import numpy as np
from demoparser2 import DemoParser
from tqdm import tqdm


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def parse_demo_folder(input_dir, cheater_ids, blacklist_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Parsing demos"):
        if not filename.endswith('.dem'):
            continue
        demo_path = os.path.join(input_dir, filename)
        demo_base = os.path.splitext(filename)[0]

        try:
            parser = DemoParser(demo_path)
            events = parser.parse_event("player_death", player=["X", "Y", "Z", "pitch", "yaw", "steamid"])
            ticks_df = parser.parse_ticks(["tick", "steamid", "X", "Y", "Z", "pitch", "yaw"])

            for _, event in events.iterrows():
                attacker = event.get("attacker_steamid")
                victim = event.get("user_steamid")
                tick = event["tick"]

                if not attacker or not victim:
                    continue

                if int(attacker) in blacklist_ids:
                    continue

                attacker_int = int(attacker)
                label = 1 if attacker_int in cheater_ids else 0

                start_tick = tick - 300
                end_tick = tick

                # Slice attacker window
                attacker_window = ticks_df[
                    ticks_df["tick"].between(start_tick, end_tick) &
                    (ticks_df["steamid"] == attacker_int)
                ].drop_duplicates(subset="tick")

                if attacker_window.empty:
                    continue

                # Reindex for consistent 300-tick window
                full_index = list(range(start_tick, end_tick))
                attacker_window = (
                    attacker_window.set_index("tick")
                    .reindex(full_index)
                    .ffill()
                    .reset_index()
                    .rename(columns={"index": "tick"})
                )

                # Add label and metadata
                attacker_window["steamid"] = attacker_int
                attacker_window["label"] = label

                # Weapon info
                weapon = event.get("weapon", "unknown").lower()
                attacker_window["weapon_name"] = weapon
                attacker_window["weapon_type"] = map_weapon_group(weapon)

                # Kill distance (Euclidean in X/Y)
                dist = euclidean_distance(
                    event.get("attacker_X", 0),
                    event.get("attacker_Y", 0),
                    event.get("user_X", 0),
                    event.get("user_Y", 0)
                )
                attacker_window["kill_distance"] = dist

                # Aim angle delta at kill (use last and second-last tick)
                if attacker_window.shape[0] >= 2:
                    pitch_delta = (
                        attacker_window["pitch"].iloc[-1] -
                        attacker_window["pitch"].iloc[-2]
                    )
                    yaw_delta = (
                        attacker_window["yaw"].iloc[-1] -
                        attacker_window["yaw"].iloc[-2]
                    )
                else:
                    pitch_delta = yaw_delta = 0

                attacker_window["pitch_delta_at_kill"] = pitch_delta
                attacker_window["yaw_delta_at_kill"] = yaw_delta

                # Player speed: Euclidean distance between last two positions
                if attacker_window.shape[0] >= 2:
                    dx = (
                        attacker_window["X"].iloc[-1] -
                        attacker_window["X"].iloc[-2]
                    )
                    dy = (
                        attacker_window["Y"].iloc[-1] -
                        attacker_window["Y"].iloc[-2]
                    )
                    dz = (
                        attacker_window["Z"].iloc[-1] -
                        attacker_window["Z"].iloc[-2]
                    )
                    speed = np.sqrt(dx**2 + dy**2 + dz**2)
                else:
                    speed = 0

                attacker_window["player_speed"] = speed

                # Save
                subfolder = "cheater" if label == 1 else "legit"
                user_dir = os.path.join(output_dir, subfolder, f"user_{attacker}")
                os.makedirs(user_dir, exist_ok=True)

                csv_name = f"{demo_base}_kill_{start_tick}_to_{end_tick}.csv"
                csv_path = os.path.join(user_dir, csv_name)
                attacker_window.to_csv(csv_path, index=False)
                print(f"Saved {csv_path}")

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

def map_weapon_group(weapon_name):
    if not isinstance(weapon_name, str):
        return "unknown"
    weapon_name = weapon_name.lower()
    if any(w in weapon_name for w in ["deagle", "glock", "usp", "p250", "tec9", "cz75", "five", "revolver"]):
        return "pistol"
    elif any(w in weapon_name for w in ["ak47", "m4a", "galil", "famas", "aug", "sg", "scar", "bizon"]):
        return "rifle"
    elif any(w in weapon_name for w in ["awp", "ssg", "scout"]):
        return "sniper"
    elif any(w in weapon_name for w in ["ump", "mac", "mp7", "mp9", "mp5"]):
        return "smg"
    elif any(w in weapon_name for w in ["m249", "negev"]):
        return "lmg"
    elif any(w in weapon_name for w in ["nova", "xm", "mag", "sawedoff"]):
        return "shotgun"
    elif any(w in weapon_name for w in ["knife", "zeus"]):
        return "melee"
    else:
        return "unknown"
    
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
        76561199101688540,
        76561199706491208,
        76561199832433151,
        76561199842141622,
        76561199634405619,
        76561199595345530,
        76561198145554926,
        76561198387960536,
        76561199849680237,
        76561199386218874,
    }

    blacklist_ids = {
        76561199548957864,
        76561198816573014,
        76561198272549077,
        76561198289804150,
        76561198110762232,
        76561199817843947,
        76561199625376224,
    }

    output_dir = "data/interim/parsed_csv"

    for folder in ["mixed", "cheater", "legit"]:
        input_dir = os.path.join("data/raw/demos", folder)
        print(f"📂 Parsing folder: {input_dir}")
        parse_demo_folder(
            input_dir=input_dir,
            cheater_ids=cheater_ids,
            blacklist_ids=blacklist_ids,
            output_dir=output_dir
        )
