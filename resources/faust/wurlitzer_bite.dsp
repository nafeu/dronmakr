declare name "Wurlitzer Bite";
declare description "Electric piano bite with a mid focus.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.002, 0.14, 0.62, 0.45, gate);
voice = os.triangle(freq) + 0.35*os.square(freq);
process = voice : fi.lowpass(1, 3800) * envelope * 0.6 <: _, _;
