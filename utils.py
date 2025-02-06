import random
import re
import uuid

BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

APP_NAME = "dronmakr"

def get_version():
    return f"{RED}┌ {APP_NAME} ■ v0.1.0 - January 2025\n│{RESET}   github.com/nafeu (phrakturemusic@proton.me)\n{RED}│{RESET}"

def with_main_prompt(text):
    return f"{RED}│{RESET}   {text}"

def with_build_preset_prompt(text):
    return f"{MAGENTA}┌ {APP_NAME} ■ preset builder {MAGENTA}┐{RESET} {text}"

def with_generate_midi_prompt(text):
    return f"{YELLOW}│{RESET}   {text}"

def with_generate_sample_prompt(text):
    return f"{GREEN}│{RESET}   {text}"

def with_stretch_sample_prompt(text):
    return f"{BLUE}│{RESET}   {text}"

def build_preset_header():
    return f"{MAGENTA}■ preset builder{RESET}"

def generate_midi_header():
    return f"{YELLOW}■ generating midi{RESET}"

def generate_sample_header():
    return f"{GREEN}■ generating sample{RESET}"

def stretch_sample_header():
    return f"{BLUE}■ stretching sample{RESET}"

def generate_id():
    return str(uuid.uuid4())[:8]

def format_name(text):
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text.replace(' ', '_').replace('-', '_').lower())
    return text

def extract_plugin(plugin_path):
    return plugin_path.split('/')[-1]


# Word lists
fantasy_prefixes = ["vael", "zyph", "eldar", "nyx", "sol", "astra", "lys", "xyra", "thal", "drith", "zeph", "kael", "myrr", "syl", "orin", "thyss", "xant", "ery", "fael", "grynn", "ithil", "jorin", "kyra", "lor", "mira", "nyth", "oryn", "phae", "quorin", "ryl", "sylva", "taur", "ulric", "vex", "wyst", "xyph", "yth", "zorin"]
fantasy_suffixes = ["ion", "ara", "eth", "is", "on", "ith", "ar", "ium", "ora", "iel", "en", "thil", "mir", "dral", "vyr", "yss", "tor", "dion", "zor", "an", "ael", "thar", "ryn", "eil", "oth", "var", "us", "er", "itha", "orra", "tiel", "aeth", "quor", "syth", "ther", "vian", "yl", "wynn", "zorin"]
sci_fi_words = ["nebula", "orion", "altair", "quasar", "andromeda", "sirius", "cosmos", "celestia", "lyra", "vega", "proxima", "epsilon", "cassiopeia", "draconis", "arcturus", "betelgeuse", "centauri", "rigel", "zeta", "polaris", "deneb", "antares", "triton", "europa", "titan", "hyperion", "callisto", "io", "ganymede", "kepler", "sagittarius", "eventide", "singularity", "voidstar", "exo", "nova", "pulsar", "zenith", "umbra", "aether", "helios", "astrolis", "stellaris", "galactica", "hypernova", "quintessa", "celestine", "eon", "xenon", "astralis", "cosmara", "nebularis", "solaris", "darkmatter", "infinity", "eclipse", "parallax", "redshift", "warp", "chronos", "asteroid", "hyperdrive", "starborn", "voidwalker"]
nature_words = ["aurora", "tide", "mist", "horizon", "gale", "echo", "drift", "ridge", "grove", "river", "summit", "meadow", "brook", "thicket", "glade", "canyon", "bluff", "prairie", "lagoon", "valley", "crest", "basin", "fjord", "cascade", "delta", "cliff", "dune", "frost", "zephyr", "monsoon", "tempest", "whisper", "drizzle", "ember", "glacier", "oasis", "serenade", "torrent", "tundra", "vortex", "zephyr", "wildwood", "brume", "mistral", "ember", "evergreen", "marsh", "cavern", "verdant", "celadon", "sierra", "breeze", "thunder", "solstice", "equinox", "eclipse", "moonbeam", "rainshadow", "starfall"]
mystic_words = ["ethereal", "void", "ancient", "obsidian", "lunar", "solar", "spectral", "phantom", "arcane", "celestial", "enigmatic", "abyssal", "runic", "mythic", "veilborn", "shadowbound", "starlit", "eldritch", "esoteric", "fabled", "otherworldly", "astral", "transcendent", "nether", "shrouded", "divine", "nocturnal", "dreamwoven", "hallowed", "chimeric", "eclipsed", "eonbound", "oracle", "mystic", "duskborn", "fey", "umbral", "liminal", "arcadian", "seraphic", "prophetic", "eldritch", "moonlit", "timeless", "whispered", "sanctified", "zephyric", "mirrored", "wraithlike"]
emotion_words = ["serene", "melancholy", "reverie", "euphoria", "solace", "enigma", "tranquil", "zeal", "wistful", "blissful", "lament", "awe", "yearning", "exultant", "poignant", "sublime", "bittersweet", "pensive", "ecstatic", "radiant", "somber", "elated", "passionate", "hushed", "mournful", "resplendent", "sorrowful", "harmonic", "contemplative", "exhilarated", "vehement", "tender", "empyreal", "rapturous", "forlorn", "delirious", "ethereal", "majestic", "bewitching", "lustrous", "enchanted", "mystified", "devout", "uplifted", "reclusive", "sentimental", "breathless", "fervent", "longing", "reposed", "unfathomable"]
atmospheric_words = ["reverberation", "resonance", "echoic", "oscillation", "harmonic", "distortion", "modulation", "overtones", "undulation", "driftwave", "pulsation", "vibration", "subharmonic", "phase", "aural", "dissonance", "convergence"]
weather_words = ["cyclone", "monsoon", "tempest", "blizzard", "downpour", "thunderhead", "mistveil", "rainfall", "frostbite", "mirage", "drizzle", "cloudburst", "whirlwind", "hailstorm", "vapor", "smog", "haze", "sundown"]
time_words = ["epoch", "eon", "dusk", "twilight", "dawn", "midnight", "stasis", "infinity", "continuum", "interlude", "aeon", "momentous", "liminal", "temporal", "vanishing", "evocation", "ephemeral", "passage"]
ancient_words = ["relic", "pantheon", "obelisk", "tomb", "oracle", "primordial", "seraph", "monolith", "ruin", "enshrined", "gilded", "ancestral", "sepulcher", "forgotten", "mythos", "omen", "summoning"]
cosmic_words = ["cybernetic", "cryosleep", "singularity", "darkmatter", "gravitation", "entanglement", "hyperspace", "astralis", "xenogenesis", "interstellar", "dimensional", "voidborn", "spacetime", "redshift", "wormhole", "chrono", "quantic"]
eerie_words = ["specter", "phantasm", "wraith", "haunting", "nocturne", "revenant", "whispering", "shrouded", "lurking", "abyss", "macabre", "unseen", "otherworldly", "eldritch", "gloaming", "chilling", "sable", "sepulchral"]

structures = [
    "{fantasy}{suffix}", # E.g., "Vaelion"
    "{sci_fi} {nature}", # E.g., "Orion Mist"
    "{emotion} {mystic}", # E.g., "Serene Void"
    "{nature} {mystic}", # E.g., "Gale Obsidian"
    "{fantasy}{suffix} {emotion}", # E.g., "Eldarion Reverie"
    "{mystic} {sci_fi} {nature}", # E.g., "Lunar Altair Drift"
    "{atmospheric} {sound}", # E.g., "Reverberation Echo"
    "{weather} {mystic}", # E.g., "Cyclone Abyssal"
    "{time} {emotion}", # E.g., "Twilight Melancholy"
    "{ancient} {mystic}", # E.g., "Oracle Arcane"
    "{sci_fi} {cosmic}", # E.g., "Singularity Void"
    "{eerie} {mystic}", # E.g., "Wraithlike Shrouded"
]

# Function to generate names
def generate_name():
    structure = random.choice(structures)
    return structure.format(
        ancient=random.choice(ancient_words),
        atmospheric=random.choice(atmospheric_words),
        cosmic=random.choice(cosmic_words),
        eerie=random.choice(eerie_words),
        emotion=random.choice(emotion_words),
        fantasy=random.choice(fantasy_prefixes),
        mystic=random.choice(mystic_words),
        nature=random.choice(nature_words),
        sci_fi=random.choice(sci_fi_words),
        sound=random.choice(atmospheric_words),
        suffix=random.choice(fantasy_suffixes),
        time=random.choice(time_words),
        weather=random.choice(weather_words),
    )