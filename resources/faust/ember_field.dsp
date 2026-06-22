declare name "Ember Field";
declare description "Crackling ember field with mid sparkle.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.06, 0.3, 0.52, 0.7, gate);
spark = no.noise : fi.resonbp(2, 2800, 4);
body = no.noise : fi.lowpass(1, 900);
process = (spark * 0.35 + body * 0.25) * envelope <: _, _;
