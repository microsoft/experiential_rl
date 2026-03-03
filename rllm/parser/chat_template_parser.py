import json
import logging
import re
from copy import deepcopy

import torch

from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

from .utils import PARSER_TEST_MESSAGES

logger = logging.getLogger(__name__)


class ChatTemplateParser:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_prompt = self._get_generation_prompt(tokenizer)

    def _get_generation_prompt(self, tokenizer):
        messages = [{"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        generation_prompt = with_prompt[len(without_prompt) :]

        return generation_prompt

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def parse_completion(self, completion_ids: list[int]):
        raise NotImplementedError("ChatTemplateParser does not support parse_completion")

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, processor=None, disable_thinking=False) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "olmo" in model_name or "olmo" in tokenizer_cls:
                logger.info(f"Using OLMoChatTemplateParser for {tokenizer.name_or_path}")
                return OLMoChatTemplateParser(tokenizer)
            elif "nanbeige" in model_name or "nanbeige" in tokenizer_cls:
                logger.info(f"Using NanbeigeChatTemplateParser for {tokenizer.name_or_path}")
                return NanbeigeChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, processor=processor, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer, processor=processor)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages):
        try:
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:last_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response = self.parse([messages[last_assistant_idx]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
        response = response[len(self.generation_prompt) :].rstrip("\n")  # handle qwen trailing newline from eot token
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_mask = [1] * len(response_ids)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages):
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            if is_asst:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                response = response[len(self.generation_prompt) :]
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([1] * len(ids))
            else:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([0] * len(ids))

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.assistant_token + "<think>\n"

        from rllm.parser.tool_parser import R1ToolParser

        self.tool_parser = R1ToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Failed to format tools: {e}")

        result = ""

        if is_first_msg:
            result += self.bos_token

        if is_first_msg and messages[0]["role"] != "system" and tools_prompt_str:
            result += self.system_token + tools_prompt_str

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message, accumulate_reasoning=False):
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not accumulate_reasoning:
            return self.assistant_token + content + self.eos_token
        elif not reasoning:
            return self.assistant_token + "<think>\n" + content + self.eos_token
        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning + "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        # Avoid mutating original message structures by parsing into a local variable
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_json = f"```json\n{json.dumps(arguments_obj)}\n```"
                        tool_call_str = f"{self.tool_parser.tool_call_begin}function{self.tool_parser.tool_sep}{tool_call_dict['name']}\n{tool_call_json}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    joined_calls_str = "\n".join(tool_calls_strs)
                    tool_calls_str = f"{self.tool_parser.tool_calls_begin}\n{joined_calls_str}\n{self.tool_parser.tool_calls_end}"
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eos_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            content = content.strip()

        tool_calls = self.tool_parser.parse(completion_text)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)

        wrapper_begin_pattern = re.escape(self.tool_parser.tool_calls_begin)
        wrapper_end_pattern = re.escape(self.tool_parser.tool_calls_end)
        content = re.sub(f"{wrapper_begin_pattern}.*?{wrapper_end_pattern}", "", content, flags=re.DOTALL)

        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, processor=None, disable_thinking=False):
        super().__init__(tokenizer, processor=processor)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\n\n</think>\n\n"
        self.generation_prompt = self.assistant_token
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        from rllm.parser.tool_parser import QwenToolParser

        self.tool_parser = QwenToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                logger.error(f"Failed to format tools: {e}")

        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tools_prompt_str + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content + self.eot_token

    def parse_user(self, message):
        if "images" in message and message["images"] is not None:
            assert isinstance(message["images"], list), "images must be a list"
            n_imgs = len(message["images"])
            content = message["content"]
            if message["content"].startswith("<image>"):
                content = content[len("<image>") :]
            vision_tokens = (self.vision_start_token + self.image_token + self.vision_end_token) * n_imgs
            return self.user_token + vision_tokens + content + self.eot_token

        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message, accumulate_reasoning=False):
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not reasoning and not tool_calls:
            return self.assistant_token + content + self.eot_token

        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning + "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_for_dump = dict(tool_call_dict)
                        if arguments_obj is not None:
                            tool_call_for_dump["arguments"] = arguments_obj
                        tool_call_str = f"{self.tool_parser.tool_call_begin}\n{json.dumps(tool_call_for_dump)}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    tool_calls_str = "\n".join(tool_calls_strs)
                except Exception as e:
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eot_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end + self.eot_token

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str + self.eot_token

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            content = content.strip()

        tool_calls = self.tool_parser.parse(content)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }

    def process_image_data(self, messages):
        from qwen_vl_utils import fetch_image

        messages = deepcopy(messages)
        image_data = []
        for message in messages:
            if "images" in message and message["images"] is not None:
                assert isinstance(message["images"], list), "images must be a list"
                images = message["images"]
                if not images or images[0] is None:
                    continue
                for image in images:
                    image_dict = image if isinstance(image, dict) else {"image": image}
                    processed_image = fetch_image(image_dict, image_patch_size=self.processor.image_processor.patch_size)  # PIL.Image.Image
                    image_data.append(processed_image)
        return image_data


class OLMoChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.eos_token = tokenizer.eos_token or ""
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        self.environment_token = "<|im_start|>environment\n"
        self.is_think_variant = "think" in str(tokenizer.name_or_path).lower()
        if self.is_think_variant:
            self.generation_prompt = self.assistant_token + "<think>"
        else:
            self.generation_prompt = self.assistant_token

    def parse(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        tools: list[Tool | dict] = None,
        use_eos: bool = True,
        **kwargs,
    ) -> str:
        tools_json = None
        if not self.is_think_variant:
            tools = tools or None
            if tools is not None:
                tool_schema_list = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_list.append(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_list.append(tool)
                    else:
                        tool_schema_list.append(tool)
                tools_json = json.dumps(tool_schema_list)

        result = ""
        has_system = any(message.get("role") == "system" for message in messages)
        if is_first_msg and not has_system:
            result += self._default_system_prompt(tools_json)

        for idx, message in enumerate(messages):
            is_last = idx == len(messages) - 1
            role = message["role"]
            if role == "system":
                result += self.parse_system(message, tools_json)
            elif role == "user":
                result += self.parse_user(message)
            elif role == "assistant":
                result += self.parse_assistant(message, is_last=is_last, use_eos=use_eos)
            elif role == "environment":
                result += self.parse_environment(message)
            elif role == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {role}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def _default_system_prompt(self, tools_json: str | None) -> str:
        if self.is_think_variant:
            return (
                self.system_token
                + "You are OLMo, a helpful function-calling AI assistant built by Ai2. "
                + "Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. "
                + "You do not currently have access to any functions. <functions></functions>"
                + self.eot_token
            )
        if tools_json is None:
            return (
                self.system_token
                + "You are a helpful function-calling AI assistant. "
                + "You do not currently have access to any functions. <functions></functions>"
                + self.eot_token
            )
        return (
            self.system_token
            + "You are a helpful function-calling AI assistant. "
            + "You are provided with function signatures within <functions></functions> XML tags. "
            + "You may call one or more functions to assist with the user query. "
            + "Output any function calls within <function_calls></function_calls> XML tags. "
            + "Do not make assumptions about what values to plug into functions."
            + "<functions>"
            + tools_json
            + "</functions>"
            + self.eot_token
        )

    def parse_system(self, message, tools_json: str | None = None):
        content = message["content"]
        if self.is_think_variant:
            if message.get("functions", None) is not None:
                content += " <functions>" + message["functions"] + "</functions>" + self.eot_token
            else:
                content += " You do not currently have access to any functions. <functions></functions>" + self.eot_token
            return self.system_token + content
        if tools_json is not None:
            content += "<functions>" + tools_json + "</functions>"
        elif message.get("functions", None) is not None:
            content += " <functions>" + message["functions"] + "</functions>"
        return self.system_token + content + self.eot_token

    def parse_user(self, message):
        if self.is_think_variant and message.get("functions", None) is not None:
            return (
                self.user_token
                + message["content"]
                + "\n<functions>"
                + message["functions"]
                + "</functions>"
                + self.eot_token
            )
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message, is_last: bool = False, use_eos: bool = True):
        content = message.get("content", None)
        if content is None:
            content = ""

        result = self.assistant_token + content
        function_calls = message.get("function_calls", None)
        tool_calls = message.get("tool_calls", None)

        if function_calls is not None:
            result += "<function_calls>" + function_calls + "</function_calls>"
        elif tool_calls is not None and not self.is_think_variant:
            result += self._format_tool_calls(tool_calls)

        if is_last and use_eos:
            result += self.eos_token
        else:
            result += self.eot_token
        return result

    def parse_environment(self, message):
        return self.environment_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.environment_token + message["content"] + self.eot_token

    def _format_tool_calls(self, tool_calls) -> str:
        formatted_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                tool_call = tool_call.to_dict()
            if isinstance(tool_call, dict) and tool_call.get("function", None) is not None:
                function = tool_call["function"]
                name = function.get("name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"arguments": arguments}
                if isinstance(arguments, dict):
                    parts = []
                    for key, value in arguments.items():
                        parts.append(f"{key}={json.dumps(value)}")
                    args_str = ", ".join(parts)
                else:
                    args_str = json.dumps(arguments)
                formatted_calls.append(f"{name}({args_str})")
            else:
                formatted_calls.append(str(tool_call))
        return "<function_calls>" + "\n".join(formatted_calls) + "</function_calls>"

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if self.eos_token and completion_text.endswith(self.eos_token):
            completion_text = completion_text[: -len(self.eos_token)]

        tool_calls = []
        content = completion_text

        match = re.search(r"<function_calls>(.*?)</function_calls>", completion_text, flags=re.DOTALL)
        if match:
            calls_text = match.group(1).strip()
            tool_calls = self._parse_function_calls(calls_text)
            content = (completion_text[: match.start()] + completion_text[match.end() :]).strip()

        content = content.strip()
        return {
            "content": content,
            "reasoning": None,
            "tool_calls": tool_calls,
        }

    def verify_equivalence(self, messages, verbose=True):
        batch_result = self.parse(messages, use_eos=True, is_first_msg=True)

        individual_results = []
        for idx, message in enumerate(messages):
            is_last = idx == len(messages) - 1
            individual_results.append(
                self.parse(
                    [message],
                    use_eos=is_last,
                    is_first_msg=(idx == 0),
                    add_generation_prompt=False,
                )
            )

        concatenated_result = "".join(individual_results)
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    def _parse_function_calls(self, text: str) -> list[ToolCall]:
        tool_calls = []
        if not text:
            return tool_calls

        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines:
            line = line.strip()
            if "(" in line and line.endswith(")"):
                name = line[: line.find("(")].strip()
                args_str = line[line.find("(") + 1 : -1].strip()
                args = self._parse_arguments(args_str)
                tool_calls.append(ToolCall(name=name, arguments=args))
            else:
                tool_calls.append(ToolCall(name=line, arguments={}))
        return tool_calls

    def _parse_arguments(self, args_str: str) -> dict:
        if not args_str:
            return {}

        parts = []
        current = []
        depth = 0
        in_string = False
        escape = False
        for ch in args_str:
            if in_string:
                current.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                current.append(ch)
                continue

            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth = max(depth - 1, 0)

            if ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(ch)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)

        args = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        return args


class NanbeigeChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.eos_token = tokenizer.eos_token or "<|im_end|>"
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        self.generation_prompt = self.assistant_token

        from rllm.parser.tool_parser import QwenToolParser

        self.tool_parser = QwenToolParser()
        self.default_system_no_tools = (
            "\u4f60\u662f\u5357\u5317\u9601\uff0c\u4e00\u6b3e\u7531BOSS\u76f4\u8058\u81ea\u4e3b\u7814\u53d1\u5e76\u8bad\u7ec3\u7684\u4e13\u4e1a\u5927\u8bed\u8a00\u6a21\u578b\u3002"
        )
        self.default_system_tools = (
            "\u4f60\u662f\u4e00\u4f4d\u5de5\u5177\u51fd\u6570\u8c03\u7528\u4e13\u5bb6\uff0c\u4f60\u4f1a\u5f97\u5230\u4e00\u4e2a\u95ee\u9898\u548c\u4e00\u7ec4\u53ef\u80fd\u7684\u5de5\u5177\u51fd\u6570\u3002\u6839\u636e\u95ee\u9898\uff0c\u4f60\u9700\u8981\u8fdb\u884c\u4e00\u4e2a\u6216\u591a\u4e2a\u51fd\u6570/\u5de5\u5177\u8c03\u7528\u4ee5\u5b9e\u73b0\u76ee\u7684\uff0c\u8bf7\u5c3d\u91cf\u5c1d\u8bd5\u63a2\u7d22\u901a\u8fc7\u5de5\u5177\u89e3\u51b3\u95ee\u9898\u3002\n"
            "\u5982\u679c\u6ca1\u6709\u4e00\u4e2a\u51fd\u6570\u53ef\u4ee5\u4f7f\u7528\uff0c\u8bf7\u76f4\u63a5\u4f7f\u7528\u81ea\u7136\u8bed\u8a00\u56de\u590d\u7528\u6237\u3002\n"
            "\u5982\u679c\u7ed9\u5b9a\u7684\u95ee\u9898\u7f3a\u5c11\u51fd\u6570\u6240\u9700\u7684\u53c2\u6570\uff0c\u8bf7\u4f7f\u7528\u81ea\u7136\u8bed\u8a00\u8fdb\u884c\u63d0\u95ee\uff0c\u5411\u7528\u6237\u8be2\u95ee\u5fc5\u8981\u4fe1\u606f\u3002\n"
            "\u5982\u679c\u8c03\u7528\u7ed3\u679c\u5df2\u7ecf\u8db3\u591f\u56de\u7b54\u7528\u6237\u95ee\u9898\uff0c\u8bf7\u5bf9\u5386\u53f2\u7ed3\u679c\u8fdb\u884c\u603b\u7ed3\uff0c\u4f7f\u7528\u81ea\u7136\u8bed\u8a00\u56de\u590d\u7528\u6237\u3002"
        )

    def parse(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        tools: list[Tool | dict] = None,
        **kwargs,
    ) -> str:
        result = ""
        tools_list = tools or []
        if is_first_msg:
            result += self._render_system_prefix(messages, tools_list)

        last_query_index = self._get_last_query_index(messages)
        for idx, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = ""

            if role in ("user", "system"):
                if role == "system" and idx == 0:
                    continue
                result += self._render_user_or_system(role, content)
            elif role == "assistant":
                result += self._render_assistant(
                    message,
                    content,
                    idx,
                    last_query_index,
                    idx == len(messages) - 1,
                )
            elif role == "tool":
                result += self._render_tool(messages, idx, content)
            else:
                raise NotImplementedError(f"Unsupported message role: {role}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def _render_system_prefix(self, messages: list[dict], tools: list) -> str:
        if tools:
            result = self.system_token
            if messages and messages[0].get("role") == "system":
                result += messages[0].get("content", "") + "\n\n"
            else:
                result += self.default_system_tools
            result += (
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>"
            )
            for tool in tools:
                if isinstance(tool, Tool):
                    tool = tool.json
                result += "\n" + json.dumps(tool)
            result += (
                "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>"
                + self.eot_token
            )
            return result

        if messages and messages[0].get("role") == "system":
            return self.system_token + messages[0].get("content", "") + self.eot_token

        return self.system_token + self.default_system_no_tools + self.eot_token

    def _render_user_or_system(self, role: str, content: str) -> str:
        return "<|im_start|>" + role + "\n" + content + self.eot_token

    def _render_assistant(self, message: dict, content: str, idx: int, last_query_index: int, is_last: bool) -> str:
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
            if "</think>" in content:
                before, _, after = content.partition("</think>")
                if "<think>" in before:
                    reasoning_content = before.split("<think>")[-1].rstrip("\n").lstrip("\n")
                content = after.lstrip("\n")

        is_after_last_query = idx > last_query_index

        if is_after_last_query:
            if is_last or reasoning_content:
                rendered = (
                    self.assistant_token
                    + "<think>\n"
                    + reasoning_content.strip("\n")
                    + "\n</think>\n\n"
                    + content.lstrip("\n")
                )
            else:
                rendered = self.assistant_token + content
        else:
            rendered = self.assistant_token + content

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            for call_idx, tool_call in enumerate(tool_calls):
                if call_idx == 0 and content:
                    rendered += "\n"
                elif call_idx > 0:
                    rendered += "\n"

                if isinstance(tool_call, ToolCall):
                    tool_call = tool_call.to_dict()
                if isinstance(tool_call, dict) and tool_call.get("function") is not None:
                    tool_call = tool_call["function"]
                name = tool_call.get("name")
                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    args_json = arguments
                else:
                    args_json = json.dumps(arguments)
                rendered += (
                    "<tool_call>\n"
                    + '{"name": "'
                    + str(name)
                    + '", "arguments": '
                    + args_json
                    + "}\n</tool_call>"
                )

        rendered += self.eot_token
        return rendered

    def _render_tool(self, messages: list[dict], idx: int, content: str) -> str:
        result = ""
        if idx == 0 or messages[idx - 1].get("role") != "tool":
            result += "<|im_start|>user"
        result += "\n<tool_response>\n" + content + "\n</tool_response>"
        if idx == len(messages) - 1 or messages[idx + 1].get("role") != "tool":
            result += self.eot_token
        return result

    def _get_last_query_index(self, messages: list[dict]) -> int:
        last_query_index = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
                continue
            last_query_index = i
            break
        return last_query_index

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        reasoning = None
        content = completion_text
        if completion_text.count("</think>") == 1:
            reasoning_part, _, remainder = completion_text.partition("</think>")
            if "<think>" in reasoning_part:
                reasoning = reasoning_part.split("<think>")[-1].strip()
            content = remainder.strip()

        tool_calls = self.tool_parser.parse(content)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL).strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # tool tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

        # TODO: add tool parser for llama

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token

    def parse_completion(self, completion_ids):
        # TODO: add parse_completion for llama
        raise NotImplementedError("LLamaChatTemplateParser does not support parse_completion")
