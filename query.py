#!/usr/bin/env python3

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def parse_range_list(value_list):
    result = []
    for item in value_list:
        if "-" in item:
            try:
                start, end = item.split("-", 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                result.extend(range(start_num, end_num + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range format: {item}")
        else:
            try:
                result.append(int(item.strip()))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid number: {item}")

    return sorted(list(set(result)))


def generate_id_map(id_map_str: str) -> Optional[Dict[str, str]]:
    if args.id_map is None:
        return None

    id_map = {}
    for id_map_str in args.id_map.split(" "):
        id, name = id_map_str.split("=")
        id_map[id] = name
    return id_map


@dataclass
class TraceEvent:
    timestamp: int
    process_id: str
    thread_id: str
    category: str
    name: str
    phase: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "pid": self.process_id,
            "tid": self.thread_id,
            "cat": self.category,
            "name": self.name,
            "ph": self.phase,
        }

    def get_identifier(self) -> Tuple[str, str, str, str]:
        """Return a unique identifier for matching B and E events"""
        return (self.process_id, self.thread_id, self.category, self.name)

from enum import IntEnum
class TsUnit(IntEnum):
    NS = 1000
    MS = 1 

class ProfileCategory(IntEnum):
    CUPTI = 1
    TRACEPOINT = 2

@dataclass
class ProfileLogFile:
    ts_unit: TsUnit
    category: ProfileCategory
    path:     str


parser = argparse.ArgumentParser(
    description="Convert the log files to chrome trace json"
)

parser.add_argument(
    "--logdir",
    help="Input log directory",
    type=str,
    required=True,
)
parser.add_argument(
    "--rank",
    help="List of worker ranks to analyze. Specify multiple ranks separated by spaces (e.g., --rank 0 1 2 3)",
    type=str,
    nargs="+",
    default=["0"],
)
"""
parser.add_argument(
    "--cupti",
    help="Query the cupti range",
    action="store_true",
)
parser.add_argument(
    "--query-step",
    help="Query step in time range",
    action="store_true",
)
"""
parser.add_argument(
    "--begin",
    type=int,
    default=0,
    help="Start time threshold in microseconds for filtering trace events. Events before this time will be excluded (default: 0)",
)
parser.add_argument(
    "--end",
    type=int,
    default=1852233025633355,
    help="End time threshold in microseconds for filtering trace events. Events after this time will be excluded (default: no limit)",
)
parser.add_argument(
    "--output",
    help="Path to save the output JSON trace file (e.g., trace.json)",
    type=str,
    default=None,
)
parser.add_argument(
    "--id-map",
    help="map the marker id to string, like --id-map 1000000000000000=train-step-it 1000000000000001=train-step-it-end",
    type=str,
    default=None,
)

args = parser.parse_args()

args.rank = parse_range_list(args.rank)
args.id_map = generate_id_map(args.id_map)

# -------------------------------------------------------------------------
# Parse rank id from a filename. 
# 
# Return rank id parsed from the filename, and
# the status of parsing.
# -------------------------------------------------------------------------
def _parse_rank_filename(filename : str):
    rank, success = -1, False
    
    if not filename.endswith('.log'):
        return rank, success
    
    # Remove .log suffix.
    filename = filename[:-4]

    # split with _
    filename_parts = filename.split('_')

    if len(filename_parts) == 2 and filename_parts[0] == 'rank' and filename_parts[1].isdigit():
        rank = int(filename_parts[1])
        success = True
        
    return rank, success

# -------------------------------------------------------------------------
# Filter profile logging records by given timestamp range.
# We use pandas chunking for better performance.
# -------------------------------------------------------------------------
import pandas as pd
def filter_profile_records_by_timestamp_range(input_file, start_ts:int, end_ts:int, ts_unit : TsUnit, chunk_line_count: int):
    
    reached_end = False
    col_names=["timestamp", "rank", "stream", "op_type", "op_name", "op_phase"]

    chunk_list = []

    for chunk in pd.read_csv(input_file, chunksize=chunk_line_count, header=None, names=col_names):

        if reached_end:
            break

        # Scale timestamp by ts unit before we query
        if ts_unit == TsUnit.NS:
            chunk['timestamp'] /= 1000

        min_ts = chunk['timestamp'].min()
        max_ts = chunk['timestamp'].max()

        if max_ts < start_ts:
            continue
        if min_ts > end_ts:
            reached_end = True
            break

        # We filter what we want in current chunk
        filtered_chunk = chunk[(chunk['timestamp'] >= start_ts) & (chunk['timestamp'] <= end_ts)]
        
        if not filtered_chunk.empty:
            chunk_list.append(filtered_chunk)

    if chunk_list:
        result_df = pd.concat(chunk_list, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Find specific profile logging files produced by CUPTI and tracepoint
# from every worker.
# 
# Return a list of relative path to logging files.
# Like 
# [
#     $(log)/sanity-check-32b-669f0566-worker-18/CUPTI/rank_152.log,
#     ...
# ]
# 
# -------------------------------------------------------------------------
def get_profile_log_files_within_ranks(logdir: Path, ranks: List[int]):
    filter_result = []

    for host_name in os.listdir(logdir):
        host_path = logdir / Path(host_name)
        
        # We only look for sub-folders under log folder.
        b_folder = os.path.isdir(host_path)
        if not b_folder:
            continue

        for profile_tool in os.listdir(host_path):

            rank_logs_path = None

            if profile_tool == 'CUPTI' or profile_tool == 'tracepoint':

                rank_logs_path = host_path / profile_tool
                # Decide the profile category
                profile_category = ProfileCategory.CUPTI if profile_tool == 'CUPTI' else ProfileCategory.TRACEPOINT
                timestamp_unit   = TsUnit.NS if profile_tool == 'CUPTI' else TsUnit.MS 
                
                for rank_log_file in os.listdir(rank_logs_path):
                    rank_log_path = rank_logs_path / rank_log_file
                    
                    rank, parse_success = _parse_rank_filename(rank_log_file)

                    if parse_success and rank in ranks:
                        # Parse timestamp in first line and last line.
                        filter_result.append(ProfileLogFile(timestamp_unit, profile_category, str(rank_log_path)))

    
    return filter_result

def generate_chrome_trace_json(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate Chrome Trace JSON structure."""
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "stackFrames": {},
        "samples": [],
    }


def generate_chrome_trace_from_logdir():
    # 1. Get rank logging files in the rank ranges.
    filtered_logs = get_profile_log_files_within_ranks(args.logdir, args.rank)

    # 2. Read each log file into a csv file. Parse first timestamp and last timestamp.
    records = []
    for filtered_log in filtered_logs:
        chunk = filter_profile_records_by_timestamp_range(filtered_log.path, args.begin, args.end, filtered_log.ts_unit, chunk_line_count=100000)
        records.append(chunk)

    all_records = pd.concat(records, ignore_index=True)

    all_events = []

    # all_records is expected to be a pandas DataFrame with columns:
    # [timestamp, rank, stream, op_type, op_name, op_phase]
    #  0,         1,    2,      3,       4,       5,
    for row in all_records.itertuples(index=False):
        # Use op_name as event name
        event_name = row[4]

        if args.id_map and "vtimeline_marker" in event_name:
            parts = event_name.split("_")
            if parts and parts[-1] in args.id_map:
                parts[-1] = args.id_map[parts[-1]]
                event_name = "_".join(parts)

        if "vtimeline_marker_begin" in event_name and row[5] == "E":
            continue
        if "vtimeline_marker_end" in event_name and row[5] == "B":
            continue

        if ("vtimeline_marker_begin" in event_name) or ("vtimeline_marker_end" in event_name):
            parts = event_name.split("_")
            # expected shape: [vtimeline, marker, begin|end, rest...]
            if len(parts) > 3:
                event_name = "_".join(parts[0:2] + parts[3:])

        # Construct a TraceEvent object.
        trace_event = TraceEvent(
            timestamp=int(row[0]),
            process_id="rank_" + str(int(row[1])),
            thread_id=("cpu" if int(row[2]) == 0 else "stream_" + str(int(row[2]))),
            category=str(row[3]),
            name=event_name,
            phase=str(row[5]),
        )

        all_events.append(trace_event.to_dict())

    return generate_chrome_trace_json(all_events)

if __name__ == "__main__":
    if args.logdir:
        trace_json = generate_chrome_trace_from_logdir()
        output_file = args.output
        if not output_file.endswith(".gz"):
            output_file += ".gz"

        print(f"Writing Chrome Trace JSON to {output_file}...")
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(trace_json, f, separators=(",", ":"))

        print(f"Open {output_file} in Chrome at https://ui.perfetto.dev/ to visualize.")

    else:
        parser.print_help()
        parser.print_usage()
