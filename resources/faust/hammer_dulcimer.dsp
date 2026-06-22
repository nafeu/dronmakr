declare name "Hammer Dulcimer";
declare description "Hammered string shimmer with harmonics.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.001, 0.65, gate);
voice = os.osc(freq) + 0.35*os.osc(freq*2.01);
process = voice * envelope * 0.65 <: _, _;
