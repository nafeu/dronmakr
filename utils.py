import random
import re
import uuid
import os
import json
import shutil

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


def with_final_auditionr_prompt(text):
    return f"{RED}│\n■{RESET} {text}"


def with_final_beatbuildr_prompt(text):
    return f"{CYAN}│\n■{RESET} {text}"


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
