import os
import sys
from typing import List

# Ensure the deletion feature is enabled for the script execution
os.environ.setdefault('ENABLE_TEST_DATA_DELETION', 'true')

from src.main import app  # noqa: E402  pylint: disable=wrong-import-position
from src.services.test_data_deletion_service import (  # noqa: E402  pylint: disable=wrong-import-position
    test_data_deletion_service,
)


def _parse_id_arg(flag: str, default: List[int]) -> List[int]:
    """
    Parse CLI args of the form --flag=1,2,3 into a list of ints.
    """
    for arg in sys.argv[1:]:
        if arg.startswith(flag + "="):
            values = arg.split("=", 1)[1]
            if not values:
                return default
            return [int(item) for item in values.split(",") if item]
    return default


def main():
    """
    Convenience entry point: wipe all dynamic data while keeping default entities.
    Usage:
        python -m src.scripts.reset_test_data
        python -m src.scripts.reset_test_data --users=1,2 --orgs=1
    """
    preserve_user_ids = _parse_id_arg("--users", [1])
    preserve_org_ids = _parse_id_arg("--orgs", [1])

    with app.app_context():
        result = test_data_deletion_service.reset_all_data(
            preserve_user_ids=preserve_user_ids,
            preserve_org_ids=preserve_org_ids,
        )
        if result:
            print(
                f"Test data reset complete. Preserved users: {preserve_user_ids}; "
                f"preserved organizations: {preserve_org_ids}"
            )
        else:
            print("Test data reset failed.")


if __name__ == "__main__":
    main()
