"""
delete_dataset.py - Script xóa dataset khỏi ClearML theo tên, project hoặc id

Cách dùng:
    python delete_dataset.py [--name <tên_dataset>] [--project <tên_project>] [--id <id_dataset>]
    [--yes]

Ví dụ:
    python scripts/delete_dataset.py --name "email-spam dataset" --project "Rain Forecast"
    python scripts/delete_dataset.py --id "1234567890abcdef"
    python scripts/delete_dataset.py --name "email-spam dataset" --project "Rain Forecast" --yes
"""

import argparse
import sys

from clearml import Dataset

try:
    from find_dataset import get_dataset_by_id
except ImportError:
    print(
        "Không import được get_dataset_by_id từ find_dataset.py. Hãy đảm bảo file này tồn tại và cùng thư mục."
    )
    sys.exit(1)

try:
    from find_dataset import list_datasets
except ImportError:
    print(
        "Không import được list_datasets từ find_dataset.py. Hãy đảm bảo file này tồn tại và cùng thư mục."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Xóa dataset khỏi ClearML theo tên, project hoặc id"
    )
    parser.add_argument("--name", help="Tên dataset cần xóa")
    parser.add_argument("--project", help="Tên project chứa dataset")
    parser.add_argument("--id", help="ID của dataset cần xóa")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Bật khớp tuyệt đối tên/project khi xóa (mặc định: khớp một phần)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Tự động xác nhận xóa (không hỏi lại)"
    )
    args = parser.parse_args()

    # Chỉ cho phép xóa theo cả name và project hoặc chỉ theo id
    if args.id:
        if args.name or args.project:
            print(
                "Không được truyền --name hoặc --project khi đã truyền --id. Chỉ chọn 1 trong 2 cách: --id hoặc --name kèm --project."
            )
            return
        ds_obj = get_dataset_by_id(args.id)
        if ds_obj is None:
            print("Không tìm thấy dataset với id đã cung cấp.")
            return
        ds_dict = {
            "id": getattr(ds_obj, "id", None),
            "name": getattr(ds_obj, "name", None),
            "project": getattr(ds_obj, "project", None),
            "version": getattr(ds_obj, "version", None),
        }
        datasets = [ds_dict]
    else:
        if not (args.name and args.project):
            print("Phải truyền đủ cả --name và --project nếu không dùng --id.")
            return
        datasets = list_datasets(name=args.name, project=args.project, exact=args.exact)

    if not datasets:
        print("Không tìm thấy dataset phù hợp để xóa.")
        return

    print("Danh sách dataset sẽ xóa:")
    for ds in datasets:
        print(
            f"  - id: {ds.get('id')}, tên: {ds.get('name')}, project: {ds.get('project')}, version: {ds.get('version', '')}"
        )

    if not args.yes:
        confirm = (
            input("Bạn có chắc chắn muốn xóa tất cả các dataset trên? (y/N): ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Đã hủy thao tác xóa.")
            return

    for ds in datasets:
        print(f"Đang xóa dataset: {ds.get('id')} - {ds.get('name')}")
        Dataset.delete(ds.get("id"))
    print("Đã xóa xong tất cả dataset phù hợp.")


if __name__ == "__main__":
    main()
