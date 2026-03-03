#!/usr/bin/env python3
import argparse
import inspect
import sys

from transformers import AutoTokenizer

from rllm.parser import ChatTemplateParser, NanbeigeChatTemplateParser, OLMoChatTemplateParser


BASE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one short sentence."},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Now say goodbye."},
]

NO_SYSTEM_MESSAGES = [
    {"role": "user", "content": "Say hello in one short sentence."},
]

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_SCHEMA_MESSAGES = [
    {"role": "user", "content": "What is the weather in Seattle?"},
]

TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Seattle?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Rainy, 9C"},
]


def _print_status(name: str, ok: bool, detail: str | None = None) -> bool:
    status = "PASS" if ok else "FAIL"
    if detail:
        print(f"[{status}] {name}: {detail}")
    else:
        print(f"[{status}] {name}")
    return ok


def _check_add_generation_prompt(tokenizer, messages):
    with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return with_prompt != without_prompt, (with_prompt, without_prompt)


def _supports_tools_arg(tokenizer) -> bool:
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    return "tools" in sig.parameters


def _get_parser(mode: str, tokenizer):
    if mode == "auto":
        return ChatTemplateParser.get_parser(tokenizer)
    if mode == "olmo":
        return OLMoChatTemplateParser(tokenizer)
    if mode == "nanbeige":
        return NanbeigeChatTemplateParser(tokenizer)
    return ChatTemplateParser(tokenizer)


def _first_diff_index(left: str, right: str) -> int | None:
    for idx, (l_ch, r_ch) in enumerate(zip(left, right)):
        if l_ch != r_ch:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if a model can use the default chat template.")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--revision", default=None, help="Optional model revision")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code")
    parser.add_argument("--with-tools", action="store_true", help="Also test tool call formatting support")
    parser.add_argument(
        "--parser",
        choices=["default", "auto", "olmo", "nanbeige"],
        default="default",
        help="Parser mode for rllm (default uses tokenizer.apply_chat_template).",
    )
    parser.add_argument(
        "--compare-parser-to-tokenizer",
        action="store_true",
        help="Compare rllm parser output to tokenizer.apply_chat_template.",
    )
    parser.add_argument("--print-prompts", action="store_true", help="Print raw prompts for inspection")
    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
    )

    ok = True
    rllm_parser = _get_parser(args.parser, tokenizer)

    # Basic chat template availability
    has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    ok &= _print_status("has_chat_template", bool(has_template), f"chat_template set={bool(has_template)}")

    # Apply chat template on base messages
    try:
        prompt = tokenizer.apply_chat_template(BASE_MESSAGES, add_generation_prompt=True, tokenize=False)
        ok &= _print_status("apply_chat_template_basic", True)
    except Exception as exc:  # noqa: BLE001
        ok &= _print_status("apply_chat_template_basic", False, str(exc))
        return 1

    # add_generation_prompt should modify output
    try:
        changed, (with_prompt, without_prompt) = _check_add_generation_prompt(tokenizer, BASE_MESSAGES)
        ok &= _print_status("add_generation_prompt_changes_output", changed)
        if args.print_prompts:
            print("\n--- with add_generation_prompt=True ---")
            print(with_prompt)
            print("\n--- with add_generation_prompt=False ---")
            print(without_prompt)
    except Exception as exc:  # noqa: BLE001
        ok &= _print_status("add_generation_prompt_changes_output", False, str(exc))

    # Compositionality check (required by rllm default parser)
    try:
        ok &= _print_status("verify_equivalence_base", rllm_parser.verify_equivalence(BASE_MESSAGES, verbose=False))
    except Exception as exc:  # noqa: BLE001
        ok &= _print_status("verify_equivalence_base", False, str(exc))

    if args.compare_parser_to_tokenizer:
        try:
            parsed_prompt = rllm_parser.parse(
                NO_SYSTEM_MESSAGES, add_generation_prompt=True, is_first_msg=True
            )
            templated_prompt = tokenizer.apply_chat_template(
                NO_SYSTEM_MESSAGES, add_generation_prompt=True, tokenize=False
            )
            matched = parsed_prompt == templated_prompt
            detail = None
            if not matched:
                diff_idx = _first_diff_index(parsed_prompt, templated_prompt)
                detail = f"first diff at index {diff_idx}" if diff_idx is not None else "prompt mismatch"
            ok &= _print_status("parser_matches_tokenizer_base", matched, detail)
            if args.print_prompts and not matched:
                print("\n--- parser prompt ---")
                print(parsed_prompt)
                print("\n--- tokenizer prompt ---")
                print(templated_prompt)
        except Exception as exc:  # noqa: BLE001
            ok &= _print_status("parser_matches_tokenizer_base", False, str(exc))

    # Optional tool support check
    if args.with_tools:
        supports_tools = _supports_tools_arg(tokenizer)
        ok &= _print_status("apply_chat_template_accepts_tools_arg", supports_tools)
        if supports_tools:
            try:
                tool_prompt = tokenizer.apply_chat_template(
                    TOOL_SCHEMA_MESSAGES, add_generation_prompt=True, tokenize=False, tools=TOOL_SCHEMA
                )
                has_tool_name = "get_weather" in tool_prompt
                ok &= _print_status("tools_appear_in_prompt", has_tool_name)
                if args.print_prompts:
                    print("\n--- tool prompt ---")
                    print(tool_prompt)
            except Exception as exc:  # noqa: BLE001
                ok &= _print_status("apply_chat_template_with_tools", False, str(exc))
        else:
            ok &= _print_status("apply_chat_template_with_tools", False, "tools arg not supported")

        if args.compare_parser_to_tokenizer and supports_tools:
            try:
                parsed_tool_prompt = rllm_parser.parse(
                    TOOL_SCHEMA_MESSAGES, add_generation_prompt=True, is_first_msg=True, tools=TOOL_SCHEMA
                )
                templated_tool_prompt = tokenizer.apply_chat_template(
                    TOOL_SCHEMA_MESSAGES, add_generation_prompt=True, tokenize=False, tools=TOOL_SCHEMA
                )
                matched = parsed_tool_prompt == templated_tool_prompt
                detail = None
                if not matched:
                    diff_idx = _first_diff_index(parsed_tool_prompt, templated_tool_prompt)
                    detail = f"first diff at index {diff_idx}" if diff_idx is not None else "prompt mismatch"
                ok &= _print_status("parser_matches_tokenizer_tools", matched, detail)
                if args.print_prompts and not matched:
                    print("\n--- parser tool prompt ---")
                    print(parsed_tool_prompt)
                    print("\n--- tokenizer tool prompt ---")
                    print(templated_tool_prompt)
            except Exception as exc:  # noqa: BLE001
                ok &= _print_status("parser_matches_tokenizer_tools", False, str(exc))

            try:
                parsed_calls_prompt = rllm_parser.parse(
                    TOOL_CALL_MESSAGES, add_generation_prompt=False, is_first_msg=True, tools=TOOL_SCHEMA
                )
                templated_calls_prompt = tokenizer.apply_chat_template(
                    TOOL_CALL_MESSAGES, add_generation_prompt=False, tokenize=False, tools=TOOL_SCHEMA
                )
                matched = parsed_calls_prompt == templated_calls_prompt
                detail = None
                if not matched:
                    diff_idx = _first_diff_index(parsed_calls_prompt, templated_calls_prompt)
                    detail = f"first diff at index {diff_idx}" if diff_idx is not None else "prompt mismatch"
                ok &= _print_status("parser_matches_tokenizer_tool_calls", matched, detail)
                if args.print_prompts and not matched:
                    print("\n--- parser tool calls prompt ---")
                    print(parsed_calls_prompt)
                    print("\n--- tokenizer tool calls prompt ---")
                    print(templated_calls_prompt)
            except Exception as exc:  # noqa: BLE001
                ok &= _print_status("parser_matches_tokenizer_tool_calls", False, str(exc))

    print("\nOverall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
