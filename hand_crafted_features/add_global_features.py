
def add_buttons(self, global_feats):

    #############################
    # 3.1 global coloring buttons
    #############################
    self.COLORS = {}
    self.add_color_button([0.60, 0.95, 0.09, 0.02], 'value', global_feats['value'])
    self.add_color_button([0.70, 0.95, 0.09, 0.02], 'actions', global_feats['actions'])
    self.add_color_button([0.60, 0.92, 0.09, 0.02], 'termination', global_feats['termination'])
    self.add_color_button([0.70, 0.92, 0.09, 0.02], 'time', global_feats['time'])
    self.add_color_button([0.60, 0.89, 0.09, 0.02], 'risk', global_feats['risk'])
    self.add_color_button([0.70, 0.89, 0.09, 0.02], 'TD', global_feats['TD'])
    self.add_color_button([0.60, 0.86, 0.09, 0.02], 'action repetition', global_feats['act_rep'])
    self.add_color_button([0.70, 0.86, 0.09, 0.02], 'reward', global_feats['reward'])

    self.SLIDER_FUNCS = []
    self.CHECK_BUTTONS = []