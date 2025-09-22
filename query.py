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


def get_rank_file_from_dir(logdir: Path) -> Dict[str, Dict[str, List[str]]]:
    result = {
        "cupti": {},
        "tracepoint": {},
    }

    for hostname in os.listdir(logdir):
        if not os.path.isdir(os.path.join(logdir, hostname)):
            continue

        if hostname not in result["cupti"]:
            result["cupti"][hostname] = []
            result["tracepoint"][hostname] = []

        for rank_file in os.listdir(os.path.join(logdir, hostname, "CUPTI")):
            cupti_rank_file = os.path.join(
                logdir,
                hostname,
                "CUPTI",
                rank_file,
            )
            result["cupti"][hostname].append(cupti_rank_file)
        for rank_file in os.listdir(os.path.join(logdir, hostname, "TracePoint")):
            tracepoint_rank_file = os.path.join(
                logdir, hostname, "TracePoint", rank_file
            )
            result["tracepoint"][hostname].append(tracepoint_rank_file)

    return result


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
    all_records = []
    # TODO: get all records from logdir

    all_events = []
    for record in all_records:
        event_name = record[5]
        if args.id_map and "vtimeline_marker" in record[5]:
            if record[5].split("_")[-1] in args.id_map:
                event_name = (
                    "_".join(record[5].split("_")[:-1])
                    + "_"
                    + args.id_map[record[5].split("_")[-1]]
                )

        if "vtimeline_marker_begin" in event_name and record[6] == "E":
            continue
        if "vtimeline_marker_end" in event_name and record[6] == "B":
            continue
        # like vtimeline_marker_begin_eventname and vtimeline_marker_end_eventname
        # remove the begin and end in the event name
        if (
            "vtimeline_marker_begin" in event_name
            or "vtimeline_marker_end" in event_name
        ):
            event_name = event_name.split("_")
            event_name = "_".join(event_name[0:2]) + "_" + "_".join(event_name[3:])

        trace_event = TraceEvent(
            timestamp=record[0],
            process_id="rank_" + str(record[2]),
            thread_id="cpu" if record[3] == 0 else "stream_" + str(record[3]),
            category=record[4],
            name=event_name,
            phase=record[6],
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
