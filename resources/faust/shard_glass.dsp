declare name "Shard Glass";
declare description "Sharp glass shard partial burst.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.1, 0.15, 0.9, gate);
voice = os.osc(freq*4.3) + os.osc(freq*6.8) + os.osc(freq*9.1);
process = voice * 0.2 * envelope <: _, _;
