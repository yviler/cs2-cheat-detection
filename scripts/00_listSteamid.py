import os
import sys
from demoparser2 import DemoParser

def list_players(demo_path):
    if not os.path.exists(demo_path):
        print(f"❌ File not found: {demo_path}")
        return

    try:
        parser = DemoParser(demo_path)
        player_info = parser.parse_player_info()
    except Exception as e:
        print(f"❌ Failed to parse player info: {e}")
        return

    if player_info.empty:
        print("⚠️ No players found in this demo.")
    else:
        print(f"\n👥 Players in demo: {os.path.basename(demo_path)}\n")
        for _, row in player_info.iterrows():
            steamid = row.get("steamid", "N/A")
            name = row.get("name", "Unknown")
            team = row.get("team_number", "-")
            print(f"- {name} — SteamID: {steamid} — Team: {team}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/00_listSteamid.py path/to/demo.dem")
    else:
        list_players(sys.argv[1])
