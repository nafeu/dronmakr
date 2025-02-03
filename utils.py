BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

APP_NAME = "dronmakr"

def with_main_prompt(text):
    return f"{RED}┌ {APP_NAME} {RED}┐{RESET} {text}"

def with_build_preset_prompt(text):
    return f"{MAGENTA}┌ {APP_NAME} ■ preset builder {MAGENTA}┐{RESET} {text}"

def with_generate_midi_prompt(text):
    return f"{YELLOW}┌ {APP_NAME} ■ midi generator {YELLOW}┐{RESET} {text}"

def with_generate_sample_prompt(text):
    return f"{GREEN}┌ {APP_NAME} ■ sample generator {GREEN}┐{RESET} {text}"

def with_stretch_sample_prompt(text):
    return f"{BLUE}┌ {APP_NAME} ■ sample stretcher {BLUE}┐{RESET} {text}"