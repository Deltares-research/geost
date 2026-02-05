import importlib
import pkgutil
import sys

PACKAGE_NAME = "geost"


def main():
    print(f"Testing imports for package: {PACKAGE_NAME}")
    package = importlib.import_module(PACKAGE_NAME)

    errors = []
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=f"{PACKAGE_NAME}."
    ):
        name = module_info.name
        try:
            importlib.import_module(name)
            print(f"Imported: {name}")
        except Exception as e:
            errors.append((name, e))
            print(f"FAILED: {name} — {e}")

    if errors:
        print("\n=== IMPORT ERRORS FOUND ===")
        for name, err in errors:
            print(f"{name}: {err}")
        sys.exit(1)

    print("\nAll modules imported successfully!")


if __name__ == "__main__":
    main()
