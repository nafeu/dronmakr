declare name "Phase Weave";
declare description "Interleaved triangle layers with motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.06, 0.25, 0.82, 0.5, gate);
voice = os.triangle(freq) + 0.7*os.triangle(freq*1.003);
process = voice * 0.5 * envelope <: _, _;
