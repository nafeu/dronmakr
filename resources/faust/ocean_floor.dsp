declare name "Ocean Floor";
declare description "Deep sine undertone with gentle noise swell.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");

envelope = gain * en.adsr(0.6, 1.0, 0.65, 2.2, gate);
tone = os.osc(freq * 0.5) * 0.7 + os.osc(freq) * 0.25;
swell = no.noise : fi.lowpass(1, 500) * 0.18 * (0.7 + 0.3 * os.lf_triangle(0.07));
process = (tone + swell) * envelope <: _, _;
