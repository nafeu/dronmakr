declare name "Prism Pad";
declare description "Shimmering harmonic prism pad.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.3, 0.55, 0.74, 1.4, gate);
voice = os.osc(freq) + 0.4*os.osc(freq*2.01) + 0.22*os.osc(freq*3.02);
process = voice * 0.38 * envelope <: _, _;
