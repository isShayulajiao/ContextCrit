import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from dataset_validation import validate_dataset


def write_json(path, data):
    """先写临时文件，再替换，避免留下不完整的JSON。"""
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as f:
            temporary = Path(f.name)
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, path)

    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON对象中存在重复键：{key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"JSON中不允许出现：{value}")


def main():
    parser = argparse.ArgumentParser(
        description="本地校验并导出通过/未通过的JSON数据"
    )
    parser.add_argument("--input", type=Path, help="数据集JSON路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="默认：输入文件旁的validation_output目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许更新已有输出文件",
    )
    args = parser.parse_args()

    if args.input is None:
        value = input("请输入数据集JSON路径：").strip().strip("\"'")
        if not value:
            raise ValueError("数据集路径不能为空")
        args.input = Path(value)

    source = args.input.expanduser().resolve()
    raw = source.read_bytes()

    samples = json.loads(
        raw.decode("utf-8-sig"),
        object_pairs_hook=strict_object,
        parse_constant=invalid_constant,
    )

    # 与一致性检验项目使用相同的本地校验规则。
    errors = validate_dataset(samples)

    valid = [
        row for row in samples
        if not errors[row["sample_id"]]
    ]
    invalid = [
        row for row in samples
        if errors[row["sample_id"]]
    ]

    directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source.parent / "validation_output"
    )

    outputs = {
        "valid": directory / f"{source.stem}_validated.json",
        "invalid": directory / f"{source.stem}_invalid.json",
        "report": directory / f"{source.stem}_validation_report.json",
    }

    # 保护原始文件，并避免无意覆盖之前的输出。
    for path in outputs.values():
        same_as_input = (
            path.resolve() == source
            or (
                path.exists()
                and os.path.samefile(path, source)
            )
        )
        if same_as_input:
            raise ValueError(f"输出与原始数据冲突，拒绝覆盖：{path}")

        if path.exists() and not args.overwrite:
            raise ValueError(
                f"输出已存在：{path}；"
                "请使用--overwrite或换--output-dir"
            )

    directory.mkdir(parents=True, exist_ok=True)

    report = {
        "input_file": str(source),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "total": len(samples),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "outputs": {
            key: str(path)
            for key, path in outputs.items()
        },
        "invalid_samples": {
            sid: reasons
            for sid, reasons in errors.items()
            if reasons
        },
    }

    write_json(outputs["invalid"], invalid)
    write_json(outputs["valid"], valid)
    write_json(outputs["report"], report)

    print(
        f"输入：{len(samples)}条；"
        f"通过：{len(valid)}条；"
        f"未通过：{len(invalid)}条"
    )

    for key, label in (
        ("valid", "通过数据"),
        ("invalid", "未通过数据"),
        ("report", "校验原因"),
    ):
        print(f"{label}：{outputs[key]}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, UnicodeError, EOFError) as exc:
        print(f"运行失败：{exc}")
        raise SystemExit(2)