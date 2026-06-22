declare name "Pipe Organ";
declare description "Pipe organ tone with strong fundamentals.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.02, 0.1, 0.92, 0.35, gate);
voice = os.osc(freq) + 0.65*os.osc(freq*2) + 0.4*os.osc(freq*3) + 0.2*os.osc(freq*4);
process = voice * 0.38 * envelope <: _, _;
