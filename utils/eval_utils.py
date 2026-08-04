import pickle

def collect_bots(start_bot, bot_directory, conn_suffix, n_per_col, bundles, bundle_size):
    counter = start_bot
    for b in range(bundles):
        all_bots = {
            "points": [],
            "springs": [],
            "weights": [],
            "ids": [],
        }
        if n_per_col > 1:
            all_bots["s_id"] = []
            all_bots["p_id"] = []
        for bot in range(counter, counter + bundle_size):
            robot_file = f"{bot_directory}/{bot}{conn_suffix}.pkl"
            with open(robot_file, "rb") as f:
                robot = pickle.load(f)
            all_bots["points"].append(robot["points"][0])
            all_bots["springs"].append(robot["springs"][0])
            all_bots["weights"].append(robot["weights"][0])
            if n_per_col == 1:
                all_bots["ids"].append(bot)
            else:
                all_bots["ids"].append(robot["group"])
            if n_per_col > 1:
                all_bots["s_id"].append(robot["s_id"][0])
                all_bots["p_id"].append(robot["p_id"][0])
        with open(f"{bot_directory}/bots_{b}.pkl", "wb") as f:
            pickle.dump(all_bots, f)
            counter += bundle_size