#!/usr/bin/env python3

import os
import json
import math
import sys
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert log files to JSON format")

parser.add_argument(
    "--input-file",
    help="Input log file path (e.g., rank_0.log)",
    type=str,
    required=True,
)
parser.add_argument(
    "--output-file",
    help="Output JSON file path (e.g., trace.json)",
    type=str,
    required=True,
)
parser.add_argument(
    "--min-time",
    type=int,
    default=0,
    help="Minimum time threshold for filtering events",
)
parser.add_argument(
    "--max-time",
    type=int,
    default=math.inf,
    help="Maximum time threshold for filtering events",
)


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


@dataclass
class MemoryDump:
    timestamp: int
    process_id: str
    memory: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "name": "GPUMem",
            "pid": self.process_id,
            "ph": "C",
            "args": {"Free": self.memory},
        }


def parse_trace_line(line: str) -> Optional[TraceEvent]:
    try:
        parts = line.strip().split(",")

        if len(parts) == 6:
            timestamp = int(parts[0])  # default is microsecond
            if timestamp == 0:
                return None
            process_id = "rank" + str(int(parts[1]))
            thread_id = "cpu"  # from cpu
            if timestamp > 1e17:  # from device
                timestamp = timestamp // 1000
                thread_id = "stream" + str(int(parts[2]))
            category = parts[3]
            name = parts[4]
            phase = parts[5]

            return TraceEvent(timestamp, process_id, thread_id, category, name, phase)
        if len(parts) == 3:
            timestamp = int(parts[0])  # default is microsecond
            if timestamp == 0:
                return None
            process_id = "rank" + str(int(parts[1]))
            memory = int(parts[2]) / 1024 / 1024
            return MemoryDump(timestamp, process_id, memory)

        return None
    except (ValueError, IndexError):
        print("parse error")
        return None


def process_trace_data(
    input_file: Path,
    min_time: Optional[int] = None,
    max_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process trace data and convert to Chrome Trace format."""
    events = []

    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            event = parse_trace_line(line)
            if not event:
                print(f"Warning: Failed to parse line {line_num}: {line}")
                continue

            if min_time is not None and event.timestamp < min_time:
                continue
            if max_time is not None and event.timestamp > max_time:
                continue
            events.append(event.to_dict())

    return events


def generate_chrome_trace_json(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate Chrome Trace JSON structure."""
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "stackFrames": {},
        "samples": [],
    }


def main():
    args = parser.parse_args()

    input_path = args.input_file
    output_file = args.output_file
    min_time = args.min_time
    max_time = args.max_time

    try:
        all_events = []

        if os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                # .*TracePoint
                if "CUPTI" in root or "TracePoint" in root:
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f" >> process file {root}/{file}")
                        all_events.extend(
                            process_trace_data(
                                file_path, min_time=min_time, max_time=max_time
                            )
                        )
        else:
            print(f" >> process file {input_path}")
            all_events.extend(
                process_trace_data(
                    input_path,
                    min_time=min_time,
                    max_time=max_time,
                )
            )

        # Generate Chrome Trace JSON
        trace_json = generate_chrome_trace_json(all_events)

        # Write to output file
        print(f"\nWriting Chrome Trace JSON to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(trace_json, f, indent=2)

        print("Successfully converted trace data!")
        print(f"Open {output_file} in Chrome at chrome://tracing/ to visualize.")

    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        print(f"Error processing trace data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
