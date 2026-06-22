declare name "Lowpass Saw";
declare description "Resonant low-pass filtered saw for mellow tones.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 1400, 120, 8000, 1);
res = hslider("res", 0.35, 0.05, 0.9, 0.01);

envelope = gain * en.adsr(0.05, 0.2, 0.8, 0.45, gate);
process = os.sawtooth(freq) : fi.resonlp(3, cutoff, res) * envelope <: _, _;
