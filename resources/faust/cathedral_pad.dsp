declare name "Cathedral Pad";
declare description "Organ-like sustained cathedral wash.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.45, 0.65, 0.78, 1.6, gate);
voice = os.osc(freq) + 0.45*os.osc(freq*2) + 0.25*os.osc(freq*3);
process = voice * 0.4 * envelope <: _, _;
