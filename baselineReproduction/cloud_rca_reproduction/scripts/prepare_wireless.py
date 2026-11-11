from pathlib import Path

import numpy as np
import pandas as pd

raw = Path("data/raw/wireless/18V_55N_Wireless")
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

alarms = pd.read_csv(raw / "Alarm.csv")
dag = np.load(raw / "DAG.npy", allow_pickle=False)

required = {
    "alarm_id", "device_id",
    "start_timestamp", "end_timestamp"
}
assert required.issubset(alarms.columns)
assert not alarms[list(required)].isna().any().any()

# Preserve the causal graph's original alarm-type indexing.
event_ids = sorted(alarms["alarm_id"].unique().tolist())
assert event_ids == list(range(dag.shape[0])), (
    f"Alarm IDs {event_ids} do not match DAG indices. "
    "Stop and investigate before training."
)

# Map physical device IDs to consecutive indices for gCastle.
device_ids = sorted(alarms["device_id"].unique().tolist())
device_to_index = {
    device_id: index
    for index, device_id in enumerate(device_ids)
}

events = pd.DataFrame({
    "event": alarms["alarm_id"].astype(int),
    "timestamp": alarms["start_timestamp"].astype(float),
    "node": alarms["device_id"].map(device_to_index).astype(int),
})

events = events.sort_values(
    ["timestamp", "node", "event"],
    kind="stable"
).reset_index(drop=True)

events.to_csv(out / "wireless_thp_events.csv", index=False)

pd.DataFrame({
    "node_index": range(len(device_ids)),
    "original_device_id": device_ids
}).to_csv(out / "wireless_device_mapping.csv", index=False)

print("=== PREPROCESSING COMPLETE ===")
print("Events:", len(events))
print("Alarm types:", len(event_ids))
print("Devices:", len(device_ids))
print("DAG shape:", dag.shape)
print("Timestamp range:",
      events["timestamp"].min(),
      events["timestamp"].max())
print(events.head().to_string(index=False))
print("Saved:", out / "wireless_thp_events.csv")
print("Saved:", out / "wireless_device_mapping.csv")
