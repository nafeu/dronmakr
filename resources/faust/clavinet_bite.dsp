declare name "Clavinet Bite";
declare description "Snappy clavinet-like filtered square.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.08, 0.55, 0.15, gate);
process = os.square(freq) : fi.lowpass(1, 3400) * envelope * 0.65 <: _, _;
