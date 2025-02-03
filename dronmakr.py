import sys

def main():
    args = sys.argv[1:]
    if not args:
        print("Working...")
        return

    command = args[0]

    if command == "-h" or command == "--help":
        print("Usage: python cli.py [options...]")
        print("\nOptions:")
        print("- -h, --help: Show this help message")
        print("- -v, --version: Display the program version")
    elif command in ("-v", "--version"):
        print("1.0.0")
    else:
        print(f"Unknown option or command: {command}. Use --help for available options.")

if __name__ == "__main__":
    main()
