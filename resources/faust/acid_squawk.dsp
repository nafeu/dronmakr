declare name "Acid Squawk";
declare description "Snappy resonant square with a squelchy filter.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 900, 120, 6000, 1);
res = hslider("res", 0.72, 0.2, 0.95, 0.01);

envelope = gain * en.adsr(0.01, 0.12, 0.65, 0.25, gate);
process = os.square(freq) : fi.resonlp(2, cutoff, res) * envelope <: _, _;
