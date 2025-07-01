#!/bin/bash

cat <<'EOF'
import triton

TOTAL_MARKER_NUM = 20
BEGIN_KERNEL_FUNCS = {}
END_KERNEL_FUNCS = {}


EOF

for i in {0..19}; do
    cat <<EOF
@triton.jit()
def vtimeline_marker_begin_$i():
    pass


@triton.jit()
def vtimeline_marker_end_$i():
    pass


vtimeline_marker_begin_$i[(1,)]()
vtimeline_marker_end_$i[(1,)]()

BEGIN_KERNEL_FUNCS[$i] = vtimeline_marker_begin_$i
END_KERNEL_FUNCS[$i] = vtimeline_marker_end_$i


EOF
done
