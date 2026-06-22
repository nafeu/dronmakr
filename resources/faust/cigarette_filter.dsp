declare name "Cigarette Filter";
declare description "Smoky low-pass triangle murmur.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 420, 80, 1800, 1);
envelope = gain * en.adsr(0.08, 0.28, 0.75, 0.45, gate);
process = os.triangle(freq) : fi.lowpass(3, cut) * envelope * 0.8 <: _, _;
