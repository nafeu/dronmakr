declare name "Abyss Rumble";
declare description "Sub-heavy drone with filtered noise undertow.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
sub = hslider("sub", 0.8, 0, 1, 0.01);

envelope = gain * en.adsr(0.5, 0.8, 0.7, 2.0, gate);
tone = os.osc(freq * 0.5) * sub + os.sawtooth(freq) * 0.35;
noiseBed = no.noise : fi.lowpass(1, 280) * 0.25;
process = (tone + noiseBed) * envelope <: _, _;
