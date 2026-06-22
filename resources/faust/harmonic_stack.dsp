declare name "Harmonic Stack";
declare description "Rich additive harmonic stack for warm tones.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.05, 0.2, 0.85, 0.4, gate);
voice = os.osc(freq) + 0.5*os.osc(freq*2) + 0.33*os.osc(freq*3) + 0.25*os.osc(freq*4);
process = voice * 0.35 * envelope <: _, _;
