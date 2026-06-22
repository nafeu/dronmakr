declare name "Tin Hat";
declare description "Bright tinny inharmonic stack.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.08, 0.12, 0.8, gate);
voice = os.osc(freq*3.2) + os.osc(freq*5.1) + os.osc(freq*7.4);
process = voice * 0.22 * envelope <: _, _;
