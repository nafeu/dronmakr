declare name "Hollow Form";
declare description "Odd-harmonic square-ish tone with space.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.05, 0.2, 0.82, 0.4, gate);
voice = os.square(freq) + 0.3*os.square(freq*3.01);
process = voice * 0.42 * envelope <: _, _;
