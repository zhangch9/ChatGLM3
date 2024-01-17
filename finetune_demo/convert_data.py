#!/usr/bin/env python


import enum
import json
from typing import Literal, Union
from pathlib import Path

import typer


class DatasetChoices(enum.Enum):
    ADGEN = 'adgen'
    TOOL_ALPACA = 'tool_alpaca'


app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)

def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']}, {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


def convert_tool_alpaca(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    raise NotImplementedError()


@app.command()
def main(dataset: DatasetChoices, data_dir: str, save_dir: str):
    """Convert a dataset to the conversation format.
    """
    if dataset == DatasetChoices.ADGEN:
        convert_fn = convert_adgen
    elif dataset == DatasetChoices.TOOL_ALPACA:
        convert_fn = convert_tool_alpaca
    else:
        raise NotImplementedError()

    convert_fn(data_dir, save_dir)



if __name__ == '__main__':
    app()