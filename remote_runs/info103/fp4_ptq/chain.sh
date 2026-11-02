#!/bin/bash
# usage: chain.sh GPU WAITPID "jobs"
while kill -0 $2 2>/dev/null; do sleep 30; done
exec ~/hyun2/fp4_ptq/rq.sh $1 "$3"
