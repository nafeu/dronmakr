declare name "Harp Glide";
declare description "Longer harp-like pluck with shimmer.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
decay = hslider("decay", 0.85, 0.2, 2.0, 0.01);
envelope = gain * en.ar(0.003, decay, gate);
process = (os.osc(freq) + 0.25*os.osc(freq*2)) * envelope * 0.65 <: _, _;
