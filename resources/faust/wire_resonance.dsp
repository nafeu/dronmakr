declare name "Wire Resonance";
declare description "Taut wire resonance with harmonic tension.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.14, 0.22, 1.2, gate);
voice = os.osc(freq*2.01) + os.osc(freq*3.97) + os.osc(freq*5.02);
process = voice * 0.28 * envelope <: _, _;
