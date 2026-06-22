declare name "Thunder Roll";
declare description "Low rolling noise swell with weight.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.35, 0.9, 0.55, 1.8, gate);
process = no.noise : fi.lowpass(2, 320) * envelope * 0.55 <: _, _;
