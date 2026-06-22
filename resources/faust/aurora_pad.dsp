declare name "Aurora Pad";
declare description "Bright triangle and saw aurora blend.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.4, 0.7, 0.7, 1.5, gate);
voice = os.triangle(freq) * 0.65 + os.sawtooth(freq) * 0.2;
process = voice : fi.lowpass(2, 4200) * envelope <: _, _;
