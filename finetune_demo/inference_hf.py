#!/usr/bin/env python
# -*- coding: utf-8 -*-


import dataclasses as dc
import functools
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import ruamel.yaml as yaml
import torch
import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TensorType,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


@dc.dataclass
class ChatMessage(object):
    role: Literal['system', 'user', 'assistant', 'observation']
    content: str
    metadata: Optional[str] = None


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


def load_model_and_tokenizer(
    model_dir: Union[str, Path], trust_remote_code: bool = False
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        # try to load as a `PeftModel`
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer


def _ensure_generation_config(
    gen_config: Optional[GenerationConfig],
    tokenizer: TokenizerType,
    chat: bool = False,
) -> GenerationConfig:
    gen_config = gen_config or GenerationConfig()
    # ensures special tokens are correctly configured
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.bos_token_id = tokenizer.bos_token_id
    gen_config.eos_token_id = [tokenizer.eos_token_id]
    if chat:
        gen_config.eos_token_id += [
            tokenizer.get_command('<|user|>'),
            tokenizer.get_command('<|observation|>'),
        ]
    return gen_config


def _generate(
    model: ModelType, input_ids: list[int], gen_config: GenerationConfig
) -> torch.Tensor:
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    output_ids = model.generate(inputs=input_ids, generation_config=gen_config)
    input_length = input_ids.size()[1]
    return output_ids[0, input_length:]


# overrides `tokenizer.apply_chat_template`
# A simplified version just for demo
def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    conversation: Conversation,
    add_generation_prompt: bool = True,
    tokenize: bool = True,
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
) -> Union[str, list[int]]:
    if not tokenize:
        raise NotImplementedError()

    input_ids = []
    for msg in conversation.messages:
        role = msg['role']
        metadata = msg['metadata'] or ''
        content = msg['content'].strip()
        encoded_ids = [
            tokenizer.get_command(f'<|{role}|>'),
            *tokenizer.tokenizer.encode(f'{metadata}\n'),
            *tokenizer.tokenizer.encode(content),
        ]
        input_ids += encoded_ids
    if add_generation_prompt:
        input_ids.append(tokenizer.get_command('<|assistant|>'))
    return tokenizer.encode(
        input_ids,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )


def _post_process(text: str) -> list[ChatMessage]:
    chat_messages = []
    messages = text.split('<|assistant|>')
    for msg in messages:
        metadata, content = map(
            lambda x: x.strip(), msg.split('\n', maxsplit=1)
        )
        if len(metadata) == 0:
            metadata = None
        chat_messages.append(
            ChatMessage('assistant', content=content, metadata=metadata)
        )
    return chat_messages


def chat_completion(
    model: ModelType,
    tokenizer: TokenizerType,
    messages: list[ChatMessage],
    gen_config: Optional[GenerationConfig] = None,
) -> list[ChatMessage]:
    conv = Conversation(messages=[dc.asdict(msg) for msg in messages])
    input_ids = apply_chat_template(
        tokenizer,
        conv,
        tokenize=True,
        add_generation_prompt=True,
    )
    gen_config = _ensure_generation_config(gen_config, tokenizer, chat=True)

    output_ids = _generate(model, input_ids, gen_config)
    text = tokenizer.decode(output_ids)
    chat_messages = _post_process(text)
    return chat_messages


def completion(
    model: ModelType,
    tokenizer: TokenizerType,
    prompt: str,
    gen_config: Optional[GenerationConfig],
) -> str:
    input_ids = tokenizer.encode(prompt)
    gen_config = _ensure_generation_config(gen_config, tokenizer)
    output_ids = _generate(model, input_ids, gen_config)
    return tokenizer.decode(output_ids)


@app.command()
def main(
    model_dir: Annotated[str, typer.Argument(help='')],
    prompt: Annotated[str, typer.Option(help='')] = 'Hi',
    trust_remote_code: Annotated[
        bool, typer.Option('--trust-remote-code', help='')
    ] = False,
    chat: Annotated[bool, typer.Option('--chat', help='')] = False,
    gen_config: Annotated[Optional[str], typer.Option(help='')] = None,
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir, trust_remote_code=trust_remote_code
    )

    if gen_config is not None:
        gen_kwargs = _get_yaml_parser().load(_resolve_path(gen_config))
        gen_config = GenerationConfig(**gen_kwargs)
    if chat:
        response = chat_completion(
            model,
            tokenizer,
            [ChatMessage('user', prompt)],
            gen_config=gen_config,
        )
    else:
        response = completion(model, tokenizer, prompt, gen_config)
    print(response)


if __name__ == '__main__':
    app()
