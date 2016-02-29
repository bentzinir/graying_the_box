import sys

sys.path.append('hand_crafted_features')

from prepare_data import prepare_data
from vis_tool import VIS_TOOL

# Parameters
run_dir = '120k'
num_frames = 5000
game_id = 1 # 0-breakout, 1-seaquest, 2-pacman
load_data = 0
debug_mode = 0

global_feats, hand_crafted_feats = prepare_data(game_id, run_dir, num_frames, load_data, debug_mode)

vis_tool = VIS_TOOL(global_feats=global_feats, hand_craft_feats=hand_crafted_feats, game_id=game_id)

vis_tool.show()