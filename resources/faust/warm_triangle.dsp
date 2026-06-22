declare name "Warm Triangle";
declare description "Soft triangle through a gentle low-pass.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 2200, 200, 10000, 1);

envelope = gain * en.adsr(0.08, 0.25, 0.82, 0.5, gate);
process = os.triangle(freq) : fi.lowpass(2, cutoff) * envelope <: _, _;
