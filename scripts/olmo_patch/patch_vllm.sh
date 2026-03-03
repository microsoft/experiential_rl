#!/usr/bin/env bash
# this is a patch script to patch vLLM to support Olmo3 tool calls

set -e

TARGET=/opt/conda/envs/rllm/lib/python3.11/site-packages/vllm/entrypoints/openai/tool_parsers

cp __init__.py "$TARGET/__init__.py"
cp olmo3_tool_parser.py "$TARGET/olmo3_tool_parser.py"

echo "vLLM Olmo3 tool calls patched."
