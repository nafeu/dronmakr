declare name "Ash Rain";
declare description "Granular rain-like noise particles.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.01, 0.55, gate);
process = no.noise : fi.highpass(1, 900) * envelope * 0.35 <: _, _;
