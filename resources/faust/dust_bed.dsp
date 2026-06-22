declare name "Dust Bed";
declare description "Low rumbling noise bed for texture layers.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 420, 80, 1200, 1);

envelope = gain * en.adsr(0.3, 0.7, 0.6, 1.5, gate);
process = no.noise : fi.lowpass(2, cutoff) * envelope * 0.45 <: _, _;
