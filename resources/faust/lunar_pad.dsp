declare name "Lunar Pad";
declare description "Slow-evolving pad with a drifting filter.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
drift = hslider("drift", 0.35, 0, 1, 0.01);
baseCut = hslider("cutoff", 1600, 300, 6000, 1);

envelope = gain * en.adsr(0.5, 0.75, 0.7, 1.8, gate);
cut = baseCut * (1 + drift * os.lf_triangle(0.09));
process = os.triangle(freq) : fi.lowpass(3, cut) * envelope * 0.8 <: _, _;
