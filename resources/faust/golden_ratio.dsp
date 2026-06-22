declare name "Golden Ratio";
declare description "Partial stack tuned to golden-ratio intervals.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.08, 0.3, 0.8, 0.55, gate);
voice = os.osc(freq) + 0.55*os.osc(freq*1.618) + 0.35*os.osc(freq*2.618);
process = voice * 0.38 * envelope <: _, _;
