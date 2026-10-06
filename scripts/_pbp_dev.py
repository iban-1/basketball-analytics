import pandas as pd

from src.bball.nba_stats import get_endpoint

pbp = get_endpoint("0042100404", "playbyplay")[0]
pd.set_option("display.width", 250, "display.max_colwidth", 110)
q = pbp[(pbp.period == 3) & (pbp.actionType.astype(str).str.contains("Turnover|Steal", case=False))]
print(q[["clock", "teamTricode", "playerNameI", "actionType", "subType", "description"]].to_string(index=False))
print(pbp.actionType.value_counts().to_dict())
s = pbp[pbp.description.astype(str).str.contains("STEAL", case=False)]
print("rows mentioning STEAL (whole game):", len(s), s.actionType.value_counts().to_dict())
