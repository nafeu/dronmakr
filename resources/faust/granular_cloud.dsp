declare name "Granular Cloud";
declare description "Tremolo noise cloud with drifting motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.2, 0.5, 0.6, 1.0, gate);
trem = 0.5 + 0.5 * os.lf_triangle(3.5);
process = no.noise : fi.lowpass(1, 2400) * envelope * trem * 0.35 <: _, _;
