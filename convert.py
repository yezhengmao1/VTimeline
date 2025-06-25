#!/usr/bin/env python3

import os
import json
import gzip
import math
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

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

    def get_identifier(self) -> Tuple[str, str, str, str]:
        """Return a unique identifier for matching B and E events"""
        return (self.process_id, self.thread_id, self.category, self.name)


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


def filter_incomplete_events(events: List[TraceEvent]) -> List[TraceEvent]:
    """Filter out events that have 'B' phase but no matching 'E' phase"""
    # Group events by their identifier
    begin_events = defaultdict(list)  # events with phase 'B'
    end_events = defaultdict(list)  # events with phase 'E'
    other_events = []  # events with other phases

    for event in events:
        if event.phase == "B":
            identifier = event.get_identifier()
            begin_events[identifier].append(event)
        elif event.phase == "E":
            identifier = event.get_identifier()
            end_events[identifier].append(event)
        else:
            other_events.append(event)

    filtered_events = []
    incomplete_count = 0

    # Process begin events and only keep those with matching end events
    for identifier, b_events in begin_events.items():
        e_events = end_events.get(identifier, [])

        # Sort events by timestamp
        b_events.sort(key=lambda x: x.timestamp)
        e_events.sort(key=lambda x: x.timestamp)

        # Match begin events with end events
        matched_pairs = 0
        min_count = min(len(b_events), len(e_events))

        # Keep matched pairs
        for i in range(min_count):
            filtered_events.append(b_events[i])
            filtered_events.append(e_events[i])
            matched_pairs += 1

        # Count incomplete events
        incomplete_count += len(b_events) - matched_pairs

        if len(b_events) != len(e_events):
            print(
                f"Warning: Event '{identifier[3]}' in {identifier[0]}/{identifier[1]} "
                f"has {len(b_events)} 'B' events but {len(e_events)} 'E' events. "
                f"Keeping {matched_pairs} matched pairs."
            )

    # Add unmatched end events (these are also incomplete)
    for identifier, e_events in end_events.items():
        if identifier not in begin_events:
            incomplete_count += len(e_events)
            print(
                f"Warning: Found {len(e_events)} 'E' events for '{identifier[3]}' "
                f"without matching 'B' events. Removing them."
            )

    # Add other events (non B/E events)
    filtered_events.extend(other_events)

    print(f"Filtered out {incomplete_count} incomplete trace events")
    return filtered_events


def process_trace_data(
    input_file: Path,
    min_time: Optional[int] = None,
    max_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process trace data and convert to Chrome Trace format."""
    trace_events = []
    memory_events = []

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

            if isinstance(event, TraceEvent):
                trace_events.append(event)
            elif isinstance(event, MemoryDump):
                memory_events.append(event)

    # Filter incomplete events if requested
    trace_events = filter_incomplete_events(trace_events)

    # Convert all events to dict
    all_events = []
    all_events.extend([event.to_dict() for event in trace_events])
    all_events.extend([event.to_dict() for event in memory_events])

    return all_events


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
        if not output_file.endswith(".gz"):
            output_file += ".gz"
        print(f"\nWriting Chrome Trace JSON to {output_file}...")
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(trace_json, f, separators=(",", ":"))

        print("Successfully converted trace data!")
        print(f"Open {output_file} in Chrome at chrome://tracing/ to visualize.")

    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        print(f"Error processing trace data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
