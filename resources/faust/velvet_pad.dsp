declare name "Velvet Pad";
declare description "Dark filtered saw pad with a long tail.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 1100, 200, 5000, 1);

envelope = gain * en.adsr(0.4, 0.55, 0.72, 1.4, gate);
process = os.sawtooth(freq) : fi.lowpass(3, cutoff) * envelope * 0.75 <: _, _;
