declare name "Vinyl Crackle";
declare description "Lo-fi crackle and hiss texture.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.08, 0.35, 0.55, 0.8, gate);
process = no.noise : fi.highpass(1, 1200) : fi.lowpass(1, 6500) * envelope * 0.4 <: _, _;
