#!/usr/bin/env python3

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import duckdb
import pandas


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
    description="Convert the log files to duckdb for analyze"
)

parser.add_argument(
    "--logdir",
    help="Input log directory",
    type=str,
    default=None,
)
parser.add_argument(
    "--db",
    help="Output duckdb file",
    type=str,
    required=True,
)
parser.add_argument(
    "--query",
    help="Query the duckdb file",
    action="store_true",
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
        print("Hostname", hostname)
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

def get_specific_rank_files_from_dir(ranks: List[int], logdir: Path) -> Dict[str, Dict[str, List[str]]]:
    result = {
        "cupti": {},
        "tracepoint": {},
    }

    target_rank_log_files = [f"rank_{rank}.log" for rank in ranks]

    for hostname in os.listdir(logdir):
        if not os.path.isdir(os.path.join(logdir, hostname)):
            continue

        if hostname not in result["cupti"]:
            result["cupti"][hostname] = []
            result["tracepoint"][hostname] = []

        cupit_path = os.path.join(logdir, hostname, "CUPTI")
        tracepoint_path = os.path.join(logdir, hostname, "TracePoint")

        for rank_file in os.listdir(cupit_path):
            if (rank_file in target_rank_log_files):
                cupti_rank_file = os.path.join(
                    cupit_path, rank_file,
                )
                result["cupti"][hostname].append(cupti_rank_file)

        for rank_file in os.listdir(tracepoint_path):
            if (rank_file in target_rank_log_files):
                tracepoint_rank_file = os.path.join(
                    tracepoint_path, rank_file
                )
                result["tracepoint"][hostname].append(tracepoint_rank_file)

    return result

def create_duckdb_table():
    if os.path.exists(f"{args.db}.duckdb"):
        raise FileExistsError(f"{args.db}.duckdb already exists")

    conn = duckdb.connect(f"{args.db}.duckdb")
    conn.execute("""
    CREATE TABLE tracepoint (
        timestamp BIGINT,
        hostname TEXT,
        rank INTEGER,
        stream INTEGER,
        op_type VARCHAR,
        op_name VARCHAR,
        op_phase VARCHAR,
    )""")
    conn.execute("CREATE INDEX idx_timestamp ON tracepoint (timestamp)")
    conn.execute("CREATE INDEX idx_rank ON tracepoint (rank)")
    conn.execute("CREATE INDEX idx_stream ON tracepoint (stream)")
    conn.execute("CREATE INDEX idx_op_type ON tracepoint (op_type)")

    return conn


def read_tracepoint_from_file(hostname: str, log_file: str, tsunit: str):
    print(f"Writing {log_file} ...")
    with open(log_file, "r") as f:
        raw_data = pandas.read_csv(
            f,
            header=None,
            names=["timestamp", "rank", "stream", "op_type", "op_name", "op_phase"],
        )
        raw_data.insert(1, "hostname", hostname)
        if tsunit == "ns":
            # convert to us
            raw_data["timestamp"] = raw_data["timestamp"] / 1000
    return raw_data


def convert_tracepoint_to_duckdb():
    log_files = {}
    
    query_all_rank_logs = (len(args.rank) == 1) and (args.rank[0] == 0)
    if (query_all_rank_logs):
        log_files = get_specific_rank_files_from_dir(args.rank, args.logdir)
    else:
        log_files = get_rank_file_from_dir(args.logdir)

    conn = create_duckdb_table()

    for hostname, rank_files in log_files["tracepoint"].items():
        for rank_file in rank_files:
            tracepoint_csv = read_tracepoint_from_file(hostname, rank_file, "us")
            conn.register("tracepoint_csv", tracepoint_csv)
            conn.execute("INSERT INTO tracepoint SELECT * FROM tracepoint_csv")
            conn.unregister("temp_csv_data")

    for hostname, rank_files in log_files["cupti"].items():
        for rank_file in rank_files:
            cupti_csv = read_tracepoint_from_file(hostname, rank_file, "ns")
            conn.register("cupti_csv_data", cupti_csv)
            conn.execute("INSERT INTO tracepoint SELECT * FROM cupti_csv_data")
            conn.unregister("cupti_csv_data")

    conn.close()

def query_cupti_from_duckdb(conn: duckdb.DuckDBPyConnection):
    
    #conn.execute(
    #    "SELECT * FROM tracepoint WHERE rank=0 and op_type='CUPTI' and (op_name='cupti-enable' or op_name='cupti-disable')"
    #)
    
    rank_filter_str = ",".join(map(str, args.rank))
    conn.execute(
        "SELECT * FROM tracepoint WHERE rank in ({}) and timestamp >= {} and timestamp <= {}".format(
            rank_filter_str,
            args.begin,
            args.end,
        )
    )

    for record in conn.fetchall():
        print(
            f"{record[5]}-{record[6]} {record[0]} <> {datetime.fromtimestamp(record[0] / 1000000).strftime('%Y-%m-%d %H:%M:%S')}"
        )


def query_step_from_duckdb(conn: duckdb.DuckDBPyConnection):
    #conn.execute(
    #    "SELECT * FROM tracepoint WHERE rank=0 and timestamp >= {} and timestamp <= {} and op_type='Train'".format(
    #        args.begin, args.end
    #    )
    #)
    rank_filter_str = ",".join(map(str, args.rank))
    conn.execute(
        "SELECT * FROM tracepoint WHERE rank in ({}) and timestamp >= {} and timestamp <= {} and op_type='Train'".format(
            rank_filter_str,
            args.begin,
            args.end,
        )
    )

    for record in conn.fetchall():
        if "train-step-it" not in record[5]:
            continue
        print(
            f"{record[5]}-{record[6]} {record[0]} <> {datetime.fromtimestamp(record[0] / 1000000).strftime('%Y-%m-%d %H:%M:%S')}"
        )


def generate_chrome_trace_json(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate Chrome Trace JSON structure."""
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "stackFrames": {},
        "samples": [],
    }


def generate_chrome_trace_from_duckdb(conn: duckdb.DuckDBPyConnection):
    rank_filter_str = ",".join(map(str, args.rank))
    conn.execute(
        "SELECT * FROM tracepoint WHERE rank in ({}) and timestamp >= {} and timestamp <= {}".format(
            rank_filter_str,
            args.begin,
            args.end,
        )
    )

    all_events = []

    for record in conn.fetchall():
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


def query_duckdb():
    conn = duckdb.connect(f"{args.db}.duckdb")

    if args.cupti:
        query_cupti_from_duckdb(conn)
    elif args.query_step:
        query_step_from_duckdb(conn)
    elif args.output:
        trace_json = generate_chrome_trace_from_duckdb(conn)
        output_file = args.output
        if not output_file.endswith(".gz"):
            output_file += ".gz"

        print(f"Writing Chrome Trace JSON to {output_file}...")
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(trace_json, f, separators=(",", ":"))

        print(f"Open {output_file} in Chrome at https://ui.perfetto.dev/ to visualize.")
    else:
        print("No query specified")
        parser.print_help()
        parser.print_usage()

    conn.close()


if __name__ == "__main__":
    if args.logdir:
        convert_tracepoint_to_duckdb()
    elif args.query:
        query_duckdb()
    else:
        parser.print_help()
        parser.print_usage()