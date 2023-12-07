"""Extract all python codeblocks from a markdown file and write them to a file."""
import pathlib
import re

import click


@click.command
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.argument(
    "output_path", type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path)
)
def main(input_path: pathlib.Path, output_path: pathlib.Path):
    markdown = input_path.read_text()

    regex = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    match = re.findall(regex, markdown, re.DOTALL | re.MULTILINE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(match))


if __name__ == "__main__":
    main()
