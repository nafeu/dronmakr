import json
import os
import random
import re
import shutil
import subprocess
import sys
import uuid

from version import __version__

BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

APP_NAME = "dronmakr"
ARCHIVE_DIR = "archive"
EXPORTS_DIR = "exports"
MIDI_DIR = "midi"
PRESETS_DIR = "presets"
PRESETS_PATH = f"{PRESETS_DIR}/presets.json"
SAVED_DIR = "saved"
TEMP_DIR = "temp"
TRASH_DIR = "trash"
PACKAGES_DIR = "packages"


def get_cli_version():
    return f"{RED}{APP_NAME} ■ v{__version__}\n{RESET}  github.com/nafeu/dronmakr (phrakturemusic@proton.me)"


def get_version():
    return f"{RED}┌ {APP_NAME} ■ v{__version__}\n│{RESET}   github.com/nafeu/dronmakr (phrakturemusic@proton.me)\n{RED}│{RESET}"


def get_auditionr_version():
    return f"{RED}┌ {APP_NAME} ■ v{__version__} auditionr\n│{RESET}   github.com/nafeu/dronmakr (phrakturemusic@proton.me){RESET}"


def get_beatbuildr_version():
    return f"{CYAN}┌ {APP_NAME} ■ v{__version__} beatbuildr\n│{RESET}   github.com/nafeu/dronmakr (phrakturemusic@proton.me){RESET}"


def with_main_prompt(text):
    return f"{RED}│{RESET}   {text}"


def with_final_main_prompt(text):
    return f"{RED}■{RESET}   {text}"


def with_build_preset_prompt(text):
    return f"{MAGENTA}┌ {APP_NAME} ■ preset builder {MAGENTA}┐{RESET} {text}"


def with_generate_drone_midi_prompt(text):
    return f"{YELLOW}│{RESET}   {text}"


def with_generate_drone_sample_prompt(text):
    return f"{GREEN}│{RESET}   {text}"


def with_process_drone_sample_prompt(text):
    return f"{BLUE}│{RESET}   {text}"


def with_generate_beat_prompt(text):
    return f"{CYAN}│{RESET}   {text}"


def build_preset_header():
    return f"{MAGENTA}┌ {APP_NAME} ■ preset builder{RESET}"


def generate_drone_midi_header():
    return f"{YELLOW}■ generating midi{RESET}"


def generate_drone_sample_header():
    return f"{GREEN}■ generating sample{RESET}"


def process_drone_sample_header():
    return f"{BLUE}■ processing sample{RESET}"


def generate_beat_header():
    return f"{CYAN}■ generating beat{RESET}"


def generate_transition_header():
    return f"{MAGENTA}■ generating transition{RESET}"


def generate_id():
    return str(uuid.uuid4())[:8]


def format_name(text):
    text = re.sub(
        r"(?<!^)(?=[A-Z])", "_", text.replace(" ", "_").replace("-", "_").lower()
    )
    return text


def extract_plugin(plugin_path):
    return plugin_path.split("/")[-1]


# Word lists
fantasy_prefixes = [
    "vael",
    "zyph",
    "eldar",
    "nyx",
    "sol",
    "astra",
    "lys",
    "xyra",
    "thal",
    "drith",
    "zeph",
    "kael",
    "myrr",
    "syl",
    "orin",
    "thyss",
    "xant",
    "ery",
    "fael",
    "grynn",
    "ithil",
    "jorin",
    "kyra",
    "lor",
    "mira",
    "nyth",
    "oryn",
    "phae",
    "quorin",
    "ryl",
    "sylva",
    "taur",
    "ulric",
    "vex",
    "wyst",
    "xyph",
    "yth",
    "zorin",
]
fantasy_suffixes = [
    "ion",
    "ara",
    "eth",
    "is",
    "on",
    "ith",
    "ar",
    "ium",
    "ora",
    "iel",
    "en",
    "thil",
    "mir",
    "dral",
    "vyr",
    "yss",
    "tor",
    "dion",
    "zor",
    "an",
    "ael",
    "thar",
    "ryn",
    "eil",
    "oth",
    "var",
    "us",
    "er",
    "itha",
    "orra",
    "tiel",
    "aeth",
    "quor",
    "syth",
    "ther",
    "vian",
    "yl",
    "wynn",
    "zorin",
]
sci_fi_words = [
    "nebula",
    "orion",
    "altair",
    "quasar",
    "andromeda",
    "sirius",
    "cosmos",
    "celestia",
    "lyra",
    "vega",
    "proxima",
    "epsilon",
    "cassiopeia",
    "draconis",
    "arcturus",
    "betelgeuse",
    "centauri",
    "rigel",
    "zeta",
    "polaris",
    "deneb",
    "antares",
    "triton",
    "europa",
    "titan",
    "hyperion",
    "callisto",
    "io",
    "ganymede",
    "kepler",
    "sagittarius",
    "eventide",
    "singularity",
    "voidstar",
    "exo",
    "nova",
    "pulsar",
    "zenith",
    "umbra",
    "aether",
    "helios",
    "astrolis",
    "stellaris",
    "galactica",
    "hypernova",
    "quintessa",
    "celestine",
    "eon",
    "xenon",
    "astralis",
    "cosmara",
    "nebularis",
    "solaris",
    "darkmatter",
    "infinity",
    "eclipse",
    "parallax",
    "redshift",
    "warp",
    "chronos",
    "asteroid",
    "hyperdrive",
    "starborn",
    "voidwalker",
]
nature_words = [
    "aurora",
    "tide",
    "mist",
    "horizon",
    "gale",
    "echo",
    "drift",
    "ridge",
    "grove",
    "river",
    "summit",
    "meadow",
    "brook",
    "thicket",
    "glade",
    "canyon",
    "bluff",
    "prairie",
    "lagoon",
    "valley",
    "crest",
    "basin",
    "fjord",
    "cascade",
    "delta",
    "cliff",
    "dune",
    "frost",
    "zephyr",
    "monsoon",
    "tempest",
    "whisper",
    "drizzle",
    "ember",
    "glacier",
    "oasis",
    "serenade",
    "torrent",
    "tundra",
    "vortex",
    "zephyr",
    "wildwood",
    "brume",
    "mistral",
    "ember",
    "evergreen",
    "marsh",
    "cavern",
    "verdant",
    "celadon",
    "sierra",
    "breeze",
    "thunder",
    "solstice",
    "equinox",
    "eclipse",
    "moonbeam",
    "rainshadow",
    "starfall",
]
mystic_words = [
    "ethereal",
    "void",
    "ancient",
    "obsidian",
    "lunar",
    "solar",
    "spectral",
    "phantom",
    "arcane",
    "celestial",
    "enigmatic",
    "abyssal",
    "runic",
    "mythic",
    "veilborn",
    "shadowbound",
    "starlit",
    "eldritch",
    "esoteric",
    "fabled",
    "otherworldly",
    "astral",
    "transcendent",
    "nether",
    "shrouded",
    "divine",
    "nocturnal",
    "dreamwoven",
    "hallowed",
    "chimeric",
    "eclipsed",
    "eonbound",
    "oracle",
    "mystic",
    "duskborn",
    "fey",
    "umbral",
    "liminal",
    "arcadian",
    "seraphic",
    "prophetic",
    "eldritch",
    "moonlit",
    "timeless",
    "whispered",
    "sanctified",
    "zephyric",
    "mirrored",
    "wraithlike",
]
emotion_words = [
    "serene",
    "melancholy",
    "reverie",
    "euphoria",
    "solace",
    "enigma",
    "tranquil",
    "zeal",
    "wistful",
    "blissful",
    "lament",
    "awe",
    "yearning",
    "exultant",
    "poignant",
    "sublime",
    "bittersweet",
    "pensive",
    "ecstatic",
    "radiant",
    "somber",
    "elated",
    "passionate",
    "hushed",
    "mournful",
    "resplendent",
    "sorrowful",
    "harmonic",
    "contemplative",
    "exhilarated",
    "vehement",
    "tender",
    "empyreal",
    "rapturous",
    "forlorn",
    "delirious",
    "ethereal",
    "majestic",
    "bewitching",
    "lustrous",
    "enchanted",
    "mystified",
    "devout",
    "uplifted",
    "reclusive",
    "sentimental",
    "breathless",
    "fervent",
    "longing",
    "reposed",
    "unfathomable",
]
atmospheric_words = [
    "reverberation",
    "resonance",
    "echoic",
    "oscillation",
    "harmonic",
    "distortion",
    "modulation",
    "overtones",
    "undulation",
    "driftwave",
    "pulsation",
    "vibration",
    "subharmonic",
    "phase",
    "aural",
    "dissonance",
    "convergence",
]
weather_words = [
    "cyclone",
    "monsoon",
    "tempest",
    "blizzard",
    "downpour",
    "thunderhead",
    "mistveil",
    "rainfall",
    "frostbite",
    "mirage",
    "drizzle",
    "cloudburst",
    "whirlwind",
    "hailstorm",
    "vapor",
    "smog",
    "haze",
    "sundown",
]
time_words = [
    "epoch",
    "eon",
    "dusk",
    "twilight",
    "dawn",
    "midnight",
    "stasis",
    "infinity",
    "continuum",
    "interlude",
    "aeon",
    "momentous",
    "liminal",
    "temporal",
    "vanishing",
    "evocation",
    "ephemeral",
    "passage",
]
ancient_words = [
    "relic",
    "pantheon",
    "obelisk",
    "tomb",
    "oracle",
    "primordial",
    "seraph",
    "monolith",
    "ruin",
    "enshrined",
    "gilded",
    "ancestral",
    "sepulcher",
    "forgotten",
    "mythos",
    "omen",
    "summoning",
]
cosmic_words = [
    "cybernetic",
    "cryosleep",
    "singularity",
    "darkmatter",
    "gravitation",
    "entanglement",
    "hyperspace",
    "astralis",
    "xenogenesis",
    "interstellar",
    "dimensional",
    "voidborn",
    "spacetime",
    "redshift",
    "wormhole",
    "chrono",
    "quantic",
]
eerie_words = [
    "specter",
    "phantasm",
    "wraith",
    "haunting",
    "nocturne",
    "revenant",
    "whispering",
    "shrouded",
    "lurking",
    "abyss",
    "macabre",
    "unseen",
    "otherworldly",
    "eldritch",
    "gloaming",
    "chilling",
    "sable",
    "sepulchral",
]

drone_structures = [
    "{fantasy}{suffix}",  # E.g., "Vaelion"
    "{sci_fi} {nature}",  # E.g., "Orion Mist"
    "{emotion} {mystic}",  # E.g., "Serene Void"
    "{nature} {mystic}",  # E.g., "Gale Obsidian"
    "{fantasy}{suffix} {emotion}",  # E.g., "Eldarion Reverie"
    "{mystic} {sci_fi} {nature}",  # E.g., "Lunar Altair Drift"
    "{atmospheric} {sound}",  # E.g., "Reverberation Echo"
    "{weather} {mystic}",  # E.g., "Cyclone Abyssal"
    "{time} {emotion}",  # E.g., "Twilight Melancholy"
    "{ancient} {mystic}",  # E.g., "Oracle Arcane"
    "{sci_fi} {cosmic}",  # E.g., "Singularity Void"
    "{eerie} {mystic}",  # E.g., "Wraithlike Shrouded"
]


# Function to generate drone names
def generate_drone_name():
    structure = random.choice(drone_structures)
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


# Word lists for beat naming
elements = [
    "hydrogen",
    "helium",
    "lithium",
    "beryllium",
    "carbon",
    "nitrogen",
    "oxygen",
    "fluorine",
    "neon",
    "sodium",
    "magnesium",
    "aluminum",
    "silicon",
    "phosphorus",
    "sulfur",
    "chlorine",
    "argon",
    "potassium",
    "calcium",
    "scandium",
    "titanium",
    "chromium",
    "manganese",
    "iron",
    "cobalt",
    "nickel",
    "copper",
    "zinc",
    "gallium",
    "germanium",
    "arsenic",
    "selenium",
    "krypton",
    "rubidium",
    "strontium",
    "yttrium",
    "zirconium",
    "niobium",
    "molybdenum",
    "technetium",
    "ruthenium",
    "rhodium",
    "palladium",
    "silver",
    "cadmium",
    "indium",
    "tin",
    "antimony",
    "tellurium",
    "iodine",
    "xenon",
    "cesium",
    "barium",
    "lanthanum",
    "cerium",
    "praseodymium",
    "neodymium",
    "promethium",
    "samarium",
    "europium",
    "gadolinium",
    "terbium",
    "dysprosium",
    "holmium",
    "erbium",
    "thulium",
    "ytterbium",
    "lutetium",
    "hafnium",
    "tantalum",
    "tungsten",
    "rhenium",
    "osmium",
    "iridium",
    "platinum",
    "gold",
    "mercury",
    "thallium",
    "lead",
    "bismuth",
    "polonium",
    "radon",
    "francium",
    "radium",
    "actinium",
    "thorium",
    "protactinium",
    "uranium",
    "neptunium",
    "plutonium",
    "americium",
]
metals = [
    "steel",
    "brass",
    "bronze",
    "chrome",
    "pewter",
    "adamant",
    "mithril",
    "alloy",
    "cobalt",
    "tungsten",
    "titanium",
    "platinum",
    "palladium",
    "iridium",
    "osmium",
    "rhodium",
    "mercury",
    "electrum",
    "solder",
    "galvanized",
    "ferrite",
    "metallic",
]
formations = [
    "crystalline",
    "matrix",
    "lattice",
    "helix",
    "fractal",
    "tessellation",
    "dodecahedron",
    "pyramid",
    "prism",
    "hexagon",
    "octagon",
    "polygon",
    "geodesic",
    "spiral",
    "vortex",
    "nexus",
    "node",
    "vertex",
    "arc",
    "axis",
    "fulcrum",
    "grid",
    "network",
    "mesh",
    "array",
    "cluster",
    "aggregate",
    "composite",
    "laminate",
    "stratum",
    "layer",
    "membrane",
    "scaffold",
    "framework",
    "structure",
    "architecture",
    "construct",
]
scientific_terms = [
    "quantum",
    "nuclear",
    "atomic",
    "molecular",
    "particle",
    "photon",
    "neutron",
    "electron",
    "proton",
    "quark",
    "lepton",
    "hadron",
    "plasma",
    "isotope",
    "catalyst",
    "enzyme",
    "synapse",
    "neuron",
    "cortex",
    "cerebral",
    "neural",
    "axon",
    "dendrite",
    "membrane",
    "cytoplasm",
    "nucleus",
    "mitochondria",
    "protein",
    "amino",
    "peptide",
    "genome",
    "chromosome",
    "mutation",
    "synthesis",
    "metabolism",
    "kinetic",
    "thermal",
    "acoustic",
    "electromagnetic",
    "gravitational",
    "magnetic",
    "electric",
    "static",
    "dynamic",
    "velocity",
    "momentum",
    "inertia",
    "friction",
    "torque",
    "tension",
    "compression",
    "elasticity",
    "viscosity",
    "entropy",
    "equilibrium",
    "resonance",
    "harmonic",
    "frequency",
    "amplitude",
    "wavelength",
    "spectrum",
    "radiation",
    "fusion",
    "fission",
    "decay",
    "collision",
    "refraction",
    "diffraction",
]
motion_forms = [
    "oscillation",
    "rotation",
    "revolution",
    "vibration",
    "pulsation",
    "undulation",
    "fluctuation",
    "perturbation",
    "gyration",
    "circulation",
    "convection",
    "diffusion",
    "propulsion",
    "trajectory",
    "acceleration",
    "deceleration",
    "momentum",
    "cascade",
    "drift",
    "shift",
    "flow",
    "surge",
    "wave",
    "ripple",
    "tremor",
    "quake",
    "shockwave",
    "impulse",
    "thrust",
    "torque",
    "spin",
    "whirl",
    "spiral",
    "helix",
    "orbit",
    "cycle",
    "rhythm",
    "cadence",
    "tempo",
    "beat",
    "pulse",
]
plants = [
    "fern",
    "moss",
    "lichen",
    "bamboo",
    "cedar",
    "cypress",
    "redwood",
    "sequoia",
    "maple",
    "oak",
    "ash",
    "elm",
    "birch",
    "willow",
    "poplar",
    "sycamore",
    "magnolia",
    "acacia",
    "eucalyptus",
    "pine",
    "spruce",
    "juniper",
    "hemlock",
    "yew",
    "lotus",
    "lily",
    "orchid",
    "jasmine",
    "lavender",
    "sage",
    "thyme",
    "basil",
    "rosemary",
    "mint",
    "chamomile",
    "ficus",
    "bonsai",
    "succulent",
    "cactus",
    "agave",
    "aloe",
    "ivy",
    "vine",
    "bramble",
    "thorn",
    "nettle",
    "reed",
    "rush",
    "sedge",
    "mangrove",
]
machinery_terms = [
    "piston",
    "valve",
    "gear",
    "turbine",
    "rotor",
    "axle",
    "crankshaft",
    "flywheel",
    "bearing",
    "sprocket",
    "pulley",
    "lever",
    "hinge",
    "spring",
    "coil",
    "bolt",
    "rivet",
    "weld",
    "forge",
    "anvil",
    "hydraulic",
    "pneumatic",
    "mechanical",
    "servo",
    "motor",
    "engine",
    "actuator",
]

beat_structures = [
    "{element} {motion}",  # E.g., "Carbon Oscillation"
    "{metal} {formation}",  # E.g., "Chrome Lattice"
    "{scientific} {motion}",  # E.g., "Quantum Pulse"
    "{formation} {time}",  # E.g., "Helix Epoch"
    "{motion} {element}",  # E.g., "Vibration Titanium"
    "{plant} {formation}",  # E.g., "Cedar Matrix"
    "{metal} {scientific}",  # E.g., "Steel Photon"
    "{element} {formation}",  # E.g., "Silicon Grid"
    "{scientific} {cosmic}",  # E.g., "Quantum Singularity"
    "{machinery} {motion}",  # E.g., "Piston Rhythm"
    "{motion} {scientific}",  # E.g., "Oscillation Neural"
    "{element} {sci_fi}",  # E.g., "Xenon Nebula"
    "{formation} {motion}",  # E.g., "Fractal Pulse"
    "{metal} {motion}",  # E.g., "Bronze Cascade"
    "{plant} {time}",  # E.g., "Bamboo Twilight"
    "{scientific} {formation}",  # E.g., "Atomic Helix"
    "{machinery} {element}",  # E.g., "Turbine Mercury"
    "{cosmic} {motion}",  # E.g., "Darkmatter Wave"
    "{element} {machinery}",  # E.g., "Iron Piston"
    "{motion} {metal}",  # E.g., "Rotation Chrome"
]


# Function to generate beat names
def generate_beat_name():
    structure = random.choice(beat_structures)
    return structure.format(
        cosmic=random.choice(cosmic_words),
        element=random.choice(elements),
        formation=random.choice(formations),
        machinery=random.choice(machinery_terms),
        metal=random.choice(metals),
        motion=random.choice(motion_forms),
        plant=random.choice(plants),
        sci_fi=random.choice(sci_fi_words),
        scientific=random.choice(scientific_terms),
        time=random.choice(time_words),
    )


def get_latest_exports(sort_override=None):
    """Reads the last 20 `.wav` files from the 'exports/' folder, sorted by newest first."""
    try:
        if not os.path.exists(EXPORTS_DIR):
            return []

        files = [
            os.path.join(EXPORTS_DIR, f)
            for f in os.listdir(EXPORTS_DIR)
            if os.path.isfile(os.path.join(EXPORTS_DIR, f))
            and f.lower().endswith(".wav")
        ]

        # Sort by newest file first (modification time descending) & return latest 5
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)[:5]

        # If sort_override is provided, remove all files in it from the final sorted list
        # and insert them at the top of the sorted list
        if sort_override:
            for file in sort_override:
                if file in sorted_files:
                    sorted_files.remove(file)
            sorted_files = sort_override + sorted_files
        return sorted_files

    except Exception as e:
        print(f"Error reading export files: {e}")
        return []


def _count_wav_in_dir(dir_path):
    """Count .wav files in a directory. Returns 0 if dir does not exist."""
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return 0
    try:
        return sum(
            1 for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(".wav")
        )
    except Exception:
        return 0


def get_auditionr_folder_counts():
    """Return counts of .wav files in archive, trash, exports, and saved folders."""
    return {
        "archive": _count_wav_in_dir(ARCHIVE_DIR),
        "trash": _count_wav_in_dir(TRASH_DIR),
        "exports": _count_wav_in_dir(EXPORTS_DIR),
        "saved": _count_wav_in_dir(SAVED_DIR),
    }


def get_presets():
    with open(PRESETS_PATH, "r") as file:
        presets = json.load(file)

    instrument_presets = []
    effect_chain_presets = []

    for preset in presets:
        if preset["type"] == "instrument":
            instrument_presets.append(preset["name"])
        elif preset["type"] == "effect_chain":
            effect_chain_presets.append(preset["name"])

    return {
        "instruments": sorted(instrument_presets),
        "effects": sorted(effect_chain_presets),
    }


def _infer_saved_sample_type(filename):
    """Infer sample type from filename for collections display. Returns one of: drone, bass, closh, drumpattern, transition, other."""
    name = filename.replace(".wav", "").lower()
    if name.startswith("drumpattern___"):
        return "drumpattern"
    if name.startswith("transition"):
        return "transition"
    if "closh" in name:
        return "closh"
    if "bass" in name or "reese" in name or "donk" in name:
        return "bass"
    if "drone" in name:
        return "drone"
    return "other"


def get_saved_files():
    """Return list of .wav files in the saved/ directory with name, path, and inferred type for collections."""
    if not os.path.exists(SAVED_DIR):
        return []
    wav_files = sorted([f for f in os.listdir(SAVED_DIR) if f.endswith(".wav")])
    return [
        {
            "name": f.replace(".wav", ""),
            "path": f"/saved/{f}",
            "type": _infer_saved_sample_type(f),
        }
        for f in wav_files
    ]


def validate_saved_paths_for_package(paths: list) -> tuple[list[dict], list[str]]:
    """
    Given URL paths like /saved/foo.wav, return entries that exist in saved/
    (same shape as get_saved_files) and a list of paths that were not allowed.
    """
    allowed = {item["path"]: item for item in get_saved_files()}
    valid: list[dict] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if not isinstance(p, str):
            invalid.append(str(p))
            continue
        key = p.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if key in allowed:
            valid.append(dict(allowed[key]))
        else:
            invalid.append(key)
    return valid, invalid


def ensure_packages_dir() -> str:
    """Create local packages/ output root (gitignored). Returns absolute path."""
    root = os.path.abspath(PACKAGES_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def parse_saved_sample_stem(stem: str) -> tuple[str, str, str]:
    """
    Split a saved sample stem (filename without .wav) into generated name, style
    segment, and trailing id segment.

    For standard names joined with ___: index 1 is generated; parts between that
    and the last segment are style/meta; the last segment is the unique id.

    For drone___name_-_chart_-_id, the tail is split on _-_.
    """
    stem = (stem or "").strip()
    parts = stem.split("___")
    if len(parts) >= 3:
        generated = parts[1]
        style = "___".join(parts[2:-1]) if len(parts) > 3 else ""
        uid = parts[-1]
        return generated, style, uid
    if len(parts) == 2 and parts[0] == "drone" and "_-_" in parts[1]:
        chunks = parts[1].split("_-_")
        if len(chunks) >= 3:
            gen = chunks[0]
            uid = chunks[-1]
            style = "_-_".join(chunks[1:-1])
            return gen, style, uid
        if len(chunks) == 2:
            return chunks[0], "", chunks[1]
    if len(parts) == 2:
        return "", parts[1], ""
    return "", stem, ""


def _normalize_name_piece(text: str) -> str:
    """Normalize one name segment and collapse any repeated underscores."""
    s = format_name((text or "").strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_package_export_basename(
    package_name: str,
    author_name: str,
    sample_index: int,
    total_selected: int,
    source_name_stem: str,
    include_generated: bool,
    include_style: bool,
) -> str:
    """
    Build destination .wav basename (no extension): package__NN__meta__author
    using double-underscore between major segments. sample_index is 1-based.
    """
    pkg = _normalize_name_piece((package_name or "").strip() or "package")
    auth = _normalize_name_piece((author_name or "").strip() or "author")
    width = max(2, len(str(max(1, total_selected))))
    num = str(sample_index).zfill(width)
    gen, style, _uid = parse_saved_sample_stem(source_name_stem)
    meta_parts: list[str] = []
    if include_generated and gen:
        g = _normalize_name_piece(gen)
        if g:
            meta_parts.append(g)
    if include_style and style:
        st = _normalize_name_piece(style)
        if st:
            meta_parts.append(st)
    sample_meta = "__".join(meta_parts) if meta_parts else ""
    chunks: list[str] = [pkg, num]
    if sample_meta:
        chunks.append(sample_meta)
    chunks.append(auth)
    return "__".join([c for c in chunks if c])


def _saved_url_to_abs(path: str) -> str | None:
    if not isinstance(path, str) or not path.startswith("/saved/"):
        return None
    rel = path[len("/saved/") :].lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    base = os.path.abspath(SAVED_DIR)
    abs_path = os.path.abspath(os.path.join(SAVED_DIR, rel))
    if not abs_path.startswith(base + os.sep) and abs_path != base:
        return None
    return abs_path if os.path.isfile(abs_path) else None


def _unique_dest_path(dest_path: str) -> str:
    if not os.path.exists(dest_path):
        return dest_path
    root, ext = os.path.splitext(dest_path)
    n = 2
    while True:
        cand = f"{root}_{n}{ext}"
        if not os.path.exists(cand):
            return cand
        n += 1


def reveal_directory_in_file_manager(dir_path: str) -> None:
    """Open a directory in the system file manager (best effort)."""
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", dir_path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", dir_path], check=False)
        else:
            subprocess.run(["xdg-open", dir_path], check=False)
    except (FileNotFoundError, OSError):
        pass


def trash_selected_saved_samples(paths_in_order: list[str]) -> dict:
    """Move selected /saved/... files to trash/. Keeps original order for reporting."""
    if not isinstance(paths_in_order, list) or not paths_in_order:
        return {"ok": False, "error": "Select at least one sample to trash."}

    allowed = {item["path"]: item for item in get_saved_files()}
    ordered: list[dict] = []
    seen: set[str] = set()
    for p in paths_in_order:
        if not isinstance(p, str):
            return {"ok": False, "error": "Each path must be a string."}
        key = p.strip()
        if key not in allowed:
            return {"ok": False, "error": f"Invalid or missing file: {key}"}
        if key in seen:
            continue
        seen.add(key)
        ordered.append(dict(allowed[key]))
    if not ordered:
        return {"ok": False, "error": "Select at least one sample to trash."}

    os.makedirs(TRASH_DIR, exist_ok=True)
    trashed: list[str] = []
    try:
        for entry in ordered:
            abs_src = _saved_url_to_abs(entry["path"])
            if not abs_src:
                return {"ok": False, "error": f"Could not resolve path: {entry['path']}"}
            base_fn = os.path.basename(abs_src)
            trash_dest = os.path.join(os.path.abspath(TRASH_DIR), base_fn)
            trash_dest = _unique_dest_path(trash_dest)
            shutil.move(abs_src, trash_dest)
            trashed.append(base_fn)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "trashed": trashed, "count": len(trashed)}


def export_collections_package(
    paths_in_order: list[str],
    package_name: str,
    author_name: str,
    include_generated: bool,
    include_style: bool,
    trash_on_save: bool,
) -> dict:
    """
    Copy selected /saved/... files into packages/{author}_-_{package}/ with
    renamed baselines. Optionally move originals to trash/ after successful copy.

    paths_in_order: URL paths in export order (1..n numbering follows this list).
    """
    pkg_name = (package_name or "").strip()
    auth_name = (author_name or "").strip()
    if not pkg_name:
        return {"ok": False, "error": "Package name is required."}
    if not auth_name:
        return {"ok": False, "error": "Author name is required."}
    if not isinstance(paths_in_order, list) or not paths_in_order:
        return {"ok": False, "error": "Select at least one sample to export."}

    allowed = {item["path"]: item for item in get_saved_files()}
    ordered: list[dict] = []
    seen: set[str] = set()
    for p in paths_in_order:
        if not isinstance(p, str):
            return {"ok": False, "error": "Each path must be a string."}
        key = p.strip()
        if key not in allowed:
            return {"ok": False, "error": f"Invalid or missing file: {key}"}
        if key in seen:
            continue
        seen.add(key)
        ordered.append(dict(allowed[key]))
    if not ordered:
        return {"ok": False, "error": "Select at least one sample to export."}

    ensure_packages_dir()
    folder_name = f"{format_name(auth_name)}_-_{format_name(pkg_name)}"
    package_dir = os.path.abspath(os.path.join(PACKAGES_DIR, folder_name))
    if os.path.exists(package_dir):
        return {
            "ok": False,
            "error": f'Package folder already exists: "{folder_name}". Choose a different name or remove the folder.',
        }

    os.makedirs(package_dir, exist_ok=True)
    total = len(ordered)
    exported_names: list[str] = []
    abs_sources: list[str] = []

    try:
        for i, entry in enumerate(ordered):
            src_url = entry["path"]
            stem = entry["name"]
            abs_src = _saved_url_to_abs(src_url)
            if not abs_src:
                raise ValueError(f"Could not resolve path: {src_url}")
            abs_sources.append(abs_src)
            base = build_package_export_basename(
                pkg_name,
                auth_name,
                i + 1,
                total,
                stem,
                include_generated,
                include_style,
            )
            dest = _unique_dest_path(os.path.join(package_dir, base + ".wav"))
            shutil.copy2(abs_src, dest)
            exported_names.append(os.path.basename(dest))

        if trash_on_save:
            os.makedirs(TRASH_DIR, exist_ok=True)
            for abs_src in abs_sources:
                base_fn = os.path.basename(abs_src)
                trash_dest = os.path.join(os.path.abspath(TRASH_DIR), base_fn)
                trash_dest = _unique_dest_path(trash_dest)
                shutil.move(abs_src, trash_dest)
    except Exception as e:
        try:
            if os.path.isdir(package_dir):
                shutil.rmtree(package_dir, ignore_errors=True)
        except Exception:
            pass
        return {"ok": False, "error": str(e)}

    reveal_directory_in_file_manager(package_dir)
    return {
        "ok": True,
        "packageDir": package_dir,
        "folderName": folder_name,
        "exported": exported_names,
        "trashOnSave": trash_on_save,
    }


def rename_samples(
    pack_name, artist_name="", affix=False, dry_run=False, delimiter="^"
):
    """Lists all .wav files in the 'saved/' directory."""
    if not os.path.exists(SAVED_DIR):
        print(f"Directory '{SAVED_DIR}' does not exist.")
        return

    if not pack_name:
        print(f"pack_name is required")
        return

    # List all .wav files
    wav_files = [f for f in os.listdir(SAVED_DIR) if f.endswith(".wav")]

    if not wav_files:
        print("No files found.")
    else:
        try:
            if affix:
                chart_list = [(file.replace(".wav", ""), file) for file in wav_files]
            else:
                chart_list = [
                    (
                        file.split(delimiter if delimiter in file else "___")[
                            0 if delimiter in file else 1
                        ],
                        file,
                    )
                    for file in wav_files
                ]
        except IndexError:
            print(f"Files have already been renamed for packaging.")
            return

        chart_list = sorted(chart_list)

        for index, chart in enumerate(chart_list):
            original_file = chart[1]
            new_name = f"{chart[0]}__{pack_name}_{index+1}{'__' + artist_name if artist_name else ''}.wav"

            if dry_run:
                print(new_name)
            else:
                original_filepath = os.path.join(SAVED_DIR, original_file)
                new_filepath = os.path.join(SAVED_DIR, new_name)
                if os.path.exists(new_filepath):
                    print(f"File '{new_filepath}' already exists, skipping renaming.")
                else:
                    os.rename(original_filepath, new_filepath)

    return wav_files


def delete_all_files(directory):
    deleted_files_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                deleted_files_count += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    return deleted_files_count
