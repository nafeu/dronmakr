import sys
import random
from generate_midi import generate_midi
from generate_sample import generate_sample
from stretch_sample import stretch_sample
from utils import with_main_prompt as with_prompt

def main():
    args = sys.argv[1:]
    if not args:
        print(with_prompt("error: missing args, use --help"))
        return

    command = args[0]

    if command == "-h" or command == "--help":
        print("Usage: python dronmakr.py [options...]")
        print("\nOptions:")
        print("- -h, --help: Show this help message")
        print("- -v, --version: Display the program version")
    elif command in ("-v", "--version"):
        print("1.0.0")
    elif command == "generate":
        if not args[1]:
            print(with_prompt("error: missing output_name"))
            return
        output_name = args[1]
        midi_style = random.choice(["chaotic_arpeggio", "chord", "split_chord", "quantized_arpeggio"])
        midi_file = generate_midi(style=midi_style)
        generated_sample = generate_sample(input_path=midi_file, output_path=f"{output_name}.wav")
        stretch_sample(input_path=generated_sample, output_path=f"{output_name}_stretched.wav")
    else:
        print(f"Unknown option or command: {command}. Use --help for available options.")

if __name__ == "__main__":
    main()
