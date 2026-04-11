"""
find_dataset.py - Script tìm kiếm dataset trên ClearML theo tên, project hoặc id

Cách dùng:
    # Tìm theo id
    python scripts/find_dataset.py --id <id>

    # Tìm theo tên và project
    python scripts/find_dataset.py --name <tên> --project <project>
    python scripts/find_dataset.py --name <tên> --project <project> --exact

    # Chỉ lấy version mới nhất (chỉ dùng với --name và --project)
    python scripts/find_dataset.py --name <tên> --project <project> --last
    python scripts/find_dataset.py --name <tên> --project <project> --last --exact

Ví dụ:
    python scripts/find_dataset.py --id "1234567890abcdef"
    python scripts/find_dataset.py --name "email-spam dataset" --project "Rain Forecast"
    python scripts/find_dataset.py --name "email-spam dataset" --project "Rain Forecast" --exact
    python scripts/find_dataset.py --name "email-spam dataset" --project "Rain Forecast" --last
    python scripts/find_dataset.py --name "email-spam dataset" --project "Rain Forecast" --last --exact
"""

import argparse

from clearml import Dataset

ARG_FIELDS = [
    {"name": "id", "kwargs": {"help": "ID của dataset cần tìm"}},
    {"name": "name", "kwargs": {"help": "Tên dataset cần tìm"}},
    {"name": "project", "kwargs": {"help": "Tên project cần tìm"}},
    {
        "name": "exact",
        "kwargs": {
            "action": "store_true",
            "help": "Bật khớp tuyệt đối (mặc định: khớp một phần)",
        },
    },
    {
        "name": "last",
        "kwargs": {
            "action": "store_true",
            "help": "Chỉ in ra dataset version mới nhất (chỉ dùng với --name và --project)",
        },
    },
]

# Biến toàn cục lưu giá trị các trường
ARG_VALUES = {}


def validate_args(args_dict):
    """
    Kiểm tra các tham số args_dict có tồn tại và hợp lệ, kiểm tra ràng buộc logic.
    Nếu thiếu field hoặc có field lạ sẽ báo lỗi tổng hợp.
    Đồng thời cập nhật ARG_VALUES.
    """
    valid_args = {arg["name"] for arg in ARG_FIELDS}
    invalid_args = [arg for arg in args_dict if arg not in valid_args]
    if invalid_args:
        raise ValueError(
            f"Tham số không hợp lệ: {', '.join('--'+f for f in invalid_args)}"
        )

    # Lưu giá trị vào biến toàn cục
    global ARG_VALUES
    ARG_VALUES = {arg: args_dict.get(arg) for arg in valid_args}

    # Kiểm tra logic ràng buộc
    if ARG_VALUES["last"] and ARG_VALUES["id"]:
        raise ValueError(
            "--last chỉ dùng khi tìm kiếm bằng --name và --project (không dùng với --id)"
        )
    if ARG_VALUES["id"]:
        if ARG_VALUES["name"] or ARG_VALUES["project"]:
            raise ValueError(
                "Không được truyền --name hoặc --project khi đã truyền --id. Chỉ chọn 1 trong 2 cách: --id hoặc --name kèm --project."
            )
        if ARG_VALUES["exact"]:
            raise ValueError(
                "--exact chỉ dùng khi tìm kiếm bằng --name và --project (không dùng với --id)"
            )
        return
    if not (ARG_VALUES["name"] and ARG_VALUES["project"]):
        raise ValueError("Phải truyền đủ cả --name và --project")


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Tìm kiếm dataset trên ClearML theo id, tên và/hoặc project, có thể chọn khớp tuyệt đối."
    )
    for field in ARG_FIELDS:
        kwargs = dict(field["kwargs"])
        parser.add_argument(f"--{field['name']}", **kwargs)
    args = parser.parse_args()
    return args


def match(dataset, id=None, name=None, project=None, exact=False):
    """
    Hàm kiểm tra dataset có khớp với tiêu chí tìm kiếm không.
    """
    if id:
        return dataset.get("id") == id
    if name and project:
        dataset_name = dataset.get("name", "")
        dataset_project = dataset.get("project", "")
        if exact:
            return dataset_name == name and dataset_project == project
        else:
            return name in dataset_name and project in dataset_project
    return False


def list_datasets(name=None, project=None, exact=False):
    """
    Trả về danh sách dataset theo tiêu chí lọc name và project.
    Nếu không truyền đủ sẽ trả về rỗng.
    """
    args_dict = {"name": name, "project": project, "exact": exact}
    validate_args(args_dict)
    all_datasets = Dataset.list_datasets() or []
    return [
        dataset
        for dataset in all_datasets
        if match(dataset, name=name, project=project, exact=exact)
    ]


def get_dataset_by_id(dataset_id):
    """Tìm dataset theo id, trả về 1 Dataset object hoặc None."""
    validate_args({"id": dataset_id})
    try:
        return Dataset.get(dataset_id=dataset_id)
    except Exception:
        return None


def find_last_version_dataset(name, project, exact=False):
    """
    Trả về dataset version mới nhất theo name và project.
    """
    datasets = list_datasets(name=name, project=project, exact=exact)
    if not datasets:
        return None

    def version_key(dataset):
        v = dataset.get("version", "")
        try:
            return float(v)
        except Exception:
            return str(v)

    datasets_sorted = sorted(datasets, key=version_key, reverse=True)
    return datasets_sorted[0]


def cli_handler():
    args = parse_cli_args()
    validate_args(vars(args))

    if args.last:
        if args.id:
            return
        last = find_last_version_dataset(args.name, args.project, args.exact)
        if last:
            print("Dataset version mới nhất:")
            print(
                f"  - id: {last.get('id')}, tên: {last.get('name')}, project: {last.get('project')}, version: {last.get('version')}"
            )
        else:
            print("Không tìm thấy dataset nào phù hợp.")
        return

    if args.id:
        dataset = get_dataset_by_id(args.id)
        if not dataset:
            print("Không tìm thấy dataset nào phù hợp.")
            return
        print("Tìm thấy dataset:")
        try:
            print(
                f"  - id: {dataset.id}, tên: {dataset.name}, project: {dataset.project}, version: {dataset.version}"
            )
        except Exception as e:
            print(f"Không thể in thông tin dataset: {e}")
        return

    results = list_datasets(name=args.name, project=args.project, exact=args.exact)

    if not results:
        print("Không tìm thấy dataset nào phù hợp.")
        return

    print(f"Tìm thấy {len(results)} dataset:")
    for dataset in results:
        print(
            f"  - id: {dataset.get('id')}, tên: {dataset.get('name')}, project: {dataset.get('project')}, version: {dataset.get('version')}"
        )

    # Nếu tìm bằng tên+project và muốn in bản mới nhất (mặc định, không dùng --last)
    if args.name and args.project and not args.id:
        last = find_last_version_dataset(args.name, args.project, args.exact)
        if last:
            print("\nDataset version mới nhất:")
            print(
                f"  - id: {last.get('id')}, tên: {last.get('name')}, project: {last.get('project')}, version: {last.get('version')}"
            )


def main():
    cli_handler()


if __name__ == "__main__":
    main()
