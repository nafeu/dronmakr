declare name "Sandstorm";
declare description "Dry granular sandstorm texture.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.12, 0.4, 0.58, 0.85, gate);
process = no.noise : fi.resonbp(2, 700, 1.8) * envelope * 0.4 <: _, _;
