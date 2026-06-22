declare name "Celeste";
declare description "Delicate celeste with a high shimmer.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.01, 0.15, 0.75, 0.8, gate);
voice = os.osc(freq) + 0.35*os.osc(freq*4.01);
process = voice * envelope * 0.55 <: _, _;
