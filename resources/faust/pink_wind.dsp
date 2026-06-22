declare name "Pink Wind";
declare description "Breathy band-pass noise for airy motion.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cutoff = hslider("cutoff", 900, 200, 4000, 1);
q = hslider("q", 1.8, 0.2, 8, 0.01);

envelope = gain * en.adsr(0.2, 0.6, 0.65, 1.0, gate);
process = no.noise : fi.resonbp(2, cutoff, q) * envelope * 0.55 <: _, _;
