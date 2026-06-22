declare name "Memory Pad";
declare description "Nostalgic pad with a very slow bloom.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.75, 1.0, 0.68, 2.0, gate);
process = os.triangle(freq) : fi.lowpass(2, 2400) * envelope * 0.75 <: _, _;
